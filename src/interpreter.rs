//! Direct evaluation of an IR `Forest`, mirroring the generated code's
//! semantics exactly: same split conditions, same f64 accumulation order,
//! same softmax algorithm. Outputs are bit-identical to compiled models,
//! which makes the interpreter both a fast `bonsai verify` engine (no rustc
//! required) and a differential-testing oracle for the code generators.
//!
//! CatBoost CTR splits are not supported: their CityHash/CTR machinery
//! lives in generated code only.

use crate::ir::{
    AggregationKind, Forest, MissingDirection, Node, Operator, PostTransform, SplitKind,
};
use anyhow::{bail, Result};

/// Predict one row. Returns one value for scalar models and `n_classes`
/// probabilities for softmax models.
pub fn predict(forest: &Forest, features: &[f32]) -> Result<Vec<f32>> {
    anyhow::ensure!(
        forest.catboost_metadata.is_none(),
        "interpreter does not support CatBoost CTR models"
    );

    if let PostTransform::Softmax { n_classes } = forest.post_transform {
        anyhow::ensure!(
            n_classes >= 2 && forest.trees.len().is_multiple_of(n_classes),
            "Tree count ({}) must be divisible by n_classes ({})",
            forest.trees.len(),
            n_classes
        );
        // Mirrors the generated softmax exactly: per-class accumulation in
        // global tree order, max-subtraction, f64 exp, cast at the end.
        let mut raw = vec![0.0f64; n_classes];
        for (c, r) in raw.iter_mut().enumerate() {
            if matches!(forest.aggregation, AggregationKind::Sum) {
                *r = if c < forest.base_scores.len() {
                    forest.base_scores[c]
                } else {
                    forest.base_score
                };
            }
        }
        for (i, tree) in forest.trees.iter().enumerate() {
            raw[i % n_classes] += eval_node(&tree.root, features)? * tree.weight;
        }
        if matches!(forest.aggregation, AggregationKind::Average) {
            let per_class = (forest.trees.len() / n_classes) as f64;
            for r in raw.iter_mut() {
                *r /= per_class;
            }
        }

        let mut max_raw = raw[0];
        for r in raw.iter().skip(1) {
            if *r > max_raw {
                max_raw = *r;
            }
        }
        let mut sum_e = 0.0f64;
        let e: Vec<f64> = raw
            .iter()
            .map(|r| {
                let v = (r - max_raw).exp();
                sum_e += v;
                v
            })
            .collect();
        return Ok(e.iter().map(|v| (v / sum_e) as f32).collect());
    }

    let raw = match forest.aggregation {
        AggregationKind::Sum => {
            let mut acc = forest.base_score;
            for tree in &forest.trees {
                acc += eval_node(&tree.root, features)? * tree.weight;
            }
            acc
        }
        AggregationKind::Average => {
            let mut acc = 0.0f64;
            for tree in &forest.trees {
                acc += eval_node(&tree.root, features)? * tree.weight;
            }
            acc / forest.trees.len() as f64
        }
    };

    let out = match forest.post_transform {
        PostTransform::Identity => raw as f32,
        PostTransform::Logit => (1.0f64 / (1.0f64 + (-raw).exp())) as f32,
        PostTransform::Log => raw.exp().min(f32::MAX as f64) as f32,
        PostTransform::Softmax { .. } => unreachable!("handled above"),
    };
    Ok(vec![out])
}

fn eval_node(node: &Node, features: &[f32]) -> Result<f64> {
    match node {
        Node::Leaf { value } => Ok(*value),
        Node::Split {
            feature_idx,
            split,
            left_child,
            right_child,
            missing_direction,
        } => {
            let val = *features.get(*feature_idx).ok_or_else(|| {
                anyhow::anyhow!(
                    "feature index {} out of bounds (row has {} features)",
                    feature_idx,
                    features.len()
                )
            })?;
            let go_left = condition(split, *missing_direction, val)?;
            eval_node(if go_left { left_child } else { right_child }, features)
        }
    }
}

/// The left-branch condition; must stay in lockstep with
/// `backends::rust::compile_condition`.
fn condition(split: &SplitKind, missing_direction: MissingDirection, val: f32) -> Result<bool> {
    if missing_direction == MissingDirection::NaVsRest {
        return Ok(!val.is_nan());
    }
    match split {
        SplitKind::Numeric {
            threshold,
            operator,
        } => {
            let cmp = match operator {
                Operator::LessThan => val < *threshold,
                Operator::LessOrEqual => val <= *threshold,
                Operator::GreaterThan => val > *threshold,
                Operator::GreaterOrEqual => val >= *threshold,
                Operator::Equal => val == *threshold,
                Operator::NotEqual => val != *threshold,
            };
            Ok(match missing_direction {
                MissingDirection::Left => val.is_nan() || cmp,
                _ => !val.is_nan() && cmp,
            })
        }
        SplitKind::Categorical {
            bitoff,
            nbits,
            data,
        } => {
            let membership = bitset_contains(*bitoff, *nbits, data, val as i32);
            Ok(match missing_direction {
                MissingDirection::Left => val.is_nan() || !membership,
                _ => !val.is_nan() && !membership,
            })
        }
        SplitKind::OnlineCtr { .. } => {
            bail!("interpreter does not support CatBoost CTR splits")
        }
    }
}

/// Identical to the generated `bitset_contains` helper.
fn bitset_contains(bitoff: u16, nbits: u32, data: &[u8], idx: i32) -> bool {
    let idx = idx - bitoff as i32;
    if idx < 0 || idx >= nbits as i32 {
        return false;
    }
    let byte_idx = (idx >> 3) as usize;
    let bit_idx = (idx & 7) as u8;
    (data[byte_idx] & (1 << bit_idx)) != 0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::rust::{generate_with_layout, Layout};
    use crate::ir::Tree;
    use crate::testutil::{compile_and_run, proba_driver, scalar_driver};

    // -----------------------------------------------------------------------
    // Semantics unit tests
    // -----------------------------------------------------------------------

    fn leaf(v: f64) -> Node {
        Node::Leaf { value: v }
    }

    fn split(
        fi: usize,
        th: f32,
        op: Operator,
        md: MissingDirection,
        left: Node,
        right: Node,
    ) -> Node {
        Node::Split {
            feature_idx: fi,
            split: SplitKind::Numeric {
                threshold: th,
                operator: op,
            },
            left_child: Box::new(left),
            right_child: Box::new(right),
            missing_direction: md,
        }
    }

    fn scalar_forest(root: Node) -> Forest {
        Forest {
            trees: vec![Tree { root, weight: 1.0 }],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            base_score: 0.0,
            base_scores: vec![],
            catboost_metadata: None,
        }
    }

    #[test]
    fn test_missing_directions() {
        let make = |md| scalar_forest(split(0, 0.5, Operator::LessThan, md, leaf(1.0), leaf(2.0)));
        let nan = [f32::NAN];
        assert_eq!(predict(&make(MissingDirection::Left), &nan).unwrap(), [1.0]);
        assert_eq!(
            predict(&make(MissingDirection::Right), &nan).unwrap(),
            [2.0]
        );
        assert_eq!(predict(&make(MissingDirection::None), &nan).unwrap(), [2.0]);
        // NaVsRest: non-NaN left, NaN right, threshold ignored.
        assert_eq!(
            predict(&make(MissingDirection::NaVsRest), &nan).unwrap(),
            [2.0]
        );
        assert_eq!(
            predict(&make(MissingDirection::NaVsRest), &[99.0]).unwrap(),
            [1.0]
        );
    }

    #[test]
    fn test_categorical_membership_goes_right() {
        // Category 2 in bitset -> right child (H2O semantics).
        let root = Node::Split {
            feature_idx: 0,
            split: SplitKind::Categorical {
                bitoff: 0,
                nbits: 8,
                data: vec![0b0000_0100],
            },
            left_child: Box::new(leaf(1.0)),
            right_child: Box::new(leaf(2.0)),
            missing_direction: MissingDirection::None,
        };
        let f = scalar_forest(root);
        assert_eq!(predict(&f, &[2.0]).unwrap(), [2.0]);
        assert_eq!(predict(&f, &[3.0]).unwrap(), [1.0]);
    }

    #[test]
    fn test_out_of_bounds_feature_errors() {
        let f = scalar_forest(split(
            5,
            0.5,
            Operator::LessThan,
            MissingDirection::None,
            leaf(1.0),
            leaf(2.0),
        ));
        assert!(predict(&f, &[0.0]).is_err());
    }

    // -----------------------------------------------------------------------
    // Differential fuzz: interpreter vs both compiled layouts, bit-for-bit
    // -----------------------------------------------------------------------

    /// Small deterministic LCG so the fuzz corpus needs no rand dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn next_usize(&mut self, bound: usize) -> usize {
            (self.next_u64() >> 33) as usize % bound
        }
        fn next_f32(&mut self, lo: f32, hi: f32) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
            lo + unit * (hi - lo)
        }
    }

    fn random_tree(rng: &mut Lcg, depth: usize, n_features: usize) -> Node {
        if depth == 0 || rng.next_usize(4) == 0 {
            return leaf(rng.next_f32(-5.0, 5.0) as f64);
        }
        let op = match rng.next_usize(6) {
            0 => Operator::LessThan,
            1 => Operator::LessOrEqual,
            2 => Operator::GreaterThan,
            3 => Operator::GreaterOrEqual,
            4 => Operator::Equal,
            _ => Operator::NotEqual,
        };
        let md = match rng.next_usize(8) {
            0 => MissingDirection::NaVsRest,
            1 | 2 => MissingDirection::Left,
            3 | 4 => MissingDirection::Right,
            _ => MissingDirection::None,
        };
        split(
            rng.next_usize(n_features),
            rng.next_f32(-2.0, 2.0),
            op,
            md,
            random_tree(rng, depth - 1, n_features),
            random_tree(rng, depth - 1, n_features),
        )
    }

    fn random_rows(rng: &mut Lcg, n_rows: usize, n_features: usize) -> Vec<Vec<f32>> {
        (0..n_rows)
            .map(|_| {
                (0..n_features)
                    .map(|_| {
                        if rng.next_usize(7) == 0 {
                            f32::NAN
                        } else {
                            rng.next_f32(-3.0, 3.0)
                        }
                    })
                    .collect()
            })
            .collect()
    }

    fn interpreter_bits(forest: &Forest, rows: &[Vec<f32>]) -> String {
        let mut out = String::new();
        for row in rows {
            for p in predict(forest, row).unwrap() {
                out.push_str(&format!("{:08x}\n", p.to_bits()));
            }
        }
        out
    }

    #[test]
    fn test_fuzz_scalar_interpreter_matches_compiled() {
        let mut rng = Lcg(42);
        for post_transform in [PostTransform::Identity, PostTransform::Logit] {
            let trees: Vec<Tree> = (0..4)
                .map(|_| Tree {
                    root: random_tree(&mut rng, 4, 5),
                    weight: rng.next_f32(0.5, 2.0) as f64,
                })
                .collect();
            let forest = Forest {
                trees,
                aggregation: AggregationKind::Sum,
                post_transform,
                base_score: rng.next_f32(-1.0, 1.0) as f64,
                base_scores: vec![],
                catboost_metadata: None,
            };
            let rows = random_rows(&mut rng, 25, 5);
            let expected = interpreter_bits(&forest, &rows);

            let driver = scalar_driver(&rows);
            for layout in [Layout::IfElse, Layout::Array] {
                let code = generate_with_layout(&forest, layout).unwrap();
                assert_eq!(
                    compile_and_run(&code, &driver),
                    expected,
                    "{:?} diverged from interpreter ({:?})",
                    layout,
                    forest.post_transform
                );
            }
        }
    }

    #[test]
    fn test_fuzz_softmax_interpreter_matches_compiled() {
        let mut rng = Lcg(7);
        let trees: Vec<Tree> = (0..6)
            .map(|_| Tree {
                root: random_tree(&mut rng, 3, 4),
                weight: 1.0,
            })
            .collect();
        let forest = Forest {
            trees,
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Softmax { n_classes: 3 },
            base_score: 0.0,
            base_scores: vec![0.1, -0.2, 0.3],
            catboost_metadata: None,
        };
        let rows = random_rows(&mut rng, 15, 4);
        let expected = interpreter_bits(&forest, &rows);

        let driver = proba_driver(&rows);
        for layout in [Layout::IfElse, Layout::Array] {
            let code = generate_with_layout(&forest, layout).unwrap();
            assert_eq!(
                compile_and_run(&code, &driver),
                expected,
                "{:?} softmax diverged from interpreter",
                layout
            );
        }
    }

    #[test]
    fn test_fuzz_categorical_interpreter_matches_ifelse() {
        let mut rng = Lcg(1234);
        // Trees mixing numeric splits with categorical bitset splits.
        fn random_cat_tree(rng: &mut Lcg, depth: usize, n_features: usize) -> Node {
            if depth == 0 || rng.next_usize(4) == 0 {
                return Node::Leaf {
                    value: rng.next_f32(-5.0, 5.0) as f64,
                };
            }
            if rng.next_usize(2) == 0 {
                let mut data = vec![0u8; 1];
                data[0] = (rng.next_u64() & 0xFF) as u8;
                Node::Split {
                    feature_idx: rng.next_usize(n_features),
                    split: SplitKind::Categorical {
                        bitoff: 0,
                        nbits: 8,
                        data,
                    },
                    left_child: Box::new(random_cat_tree(rng, depth - 1, n_features)),
                    right_child: Box::new(random_cat_tree(rng, depth - 1, n_features)),
                    missing_direction: if rng.next_usize(2) == 0 {
                        MissingDirection::Left
                    } else {
                        MissingDirection::None
                    },
                }
            } else {
                Node::Split {
                    feature_idx: rng.next_usize(n_features),
                    split: SplitKind::Numeric {
                        threshold: rng.next_f32(0.0, 8.0),
                        operator: Operator::LessThan,
                    },
                    left_child: Box::new(random_cat_tree(rng, depth - 1, n_features)),
                    right_child: Box::new(random_cat_tree(rng, depth - 1, n_features)),
                    missing_direction: MissingDirection::Right,
                }
            }
        }

        let trees: Vec<Tree> = (0..3)
            .map(|_| Tree {
                root: random_cat_tree(&mut rng, 3, 3),
                weight: 1.0,
            })
            .collect();
        let forest = Forest {
            trees,
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            base_score: 0.0,
            base_scores: vec![],
            catboost_metadata: None,
        };
        // Category-like feature values (small integers), some NaN.
        let rows: Vec<Vec<f32>> = (0..20)
            .map(|_| {
                (0..3)
                    .map(|_| {
                        if rng.next_usize(8) == 0 {
                            f32::NAN
                        } else {
                            rng.next_usize(8) as f32
                        }
                    })
                    .collect()
            })
            .collect();
        let expected = interpreter_bits(&forest, &rows);
        let code = generate_with_layout(&forest, Layout::IfElse).unwrap();
        assert_eq!(compile_and_run(&code, &scalar_driver(&rows)), expected);
    }
}
