//! Array-layout code generation.
//!
//! Flattens the forest into static parallel arrays walked by a small loop,
//! instead of one nested if/else function per tree. Generated source size and
//! rustc compile time stay roughly linear in the number of nodes with a small
//! constant, which keeps very large forests (hundreds of thousands of nodes)
//! practical to compile.
//!
//! Scope: numeric splits only. Forests with categorical bitsets or CatBoost
//! CTR features are rejected by `rust::resolve_layout` before reaching here.

use crate::ir::{AggregationKind, Forest, MissingDirection, Node, PostTransform, SplitKind, Tree};
use anyhow::{bail, Result};
use proc_macro2::TokenStream;
use quote::quote;

use super::rust::{exp_expr, f32_token, f64_token, post_transform_expr};

/// FLAGS bit layout: bits 0-2 = operator, bit 3 = NaN goes left, bit 4 = NaVsRest.
const OP_MASK: u8 = 0b0000_0111;
const FLAG_MISSING_LEFT: u8 = 0b0000_1000;
const FLAG_NA_VS_REST: u8 = 0b0001_0000;

/// Operator codes stored in the low FLAGS bits.
const OP_LT: u8 = 0;
const OP_LE: u8 = 1;
const OP_GT: u8 = 2;
const OP_GE: u8 = 3;
const OP_EQ: u8 = 4;
const OP_NE: u8 = 5;

/// The forest flattened into parallel arrays. Child links are `i32`: a
/// non-negative value is a node index, a negative value `v` refers to leaf
/// `!v` in the leaves array.
#[derive(Default)]
struct FlatForest {
    feat: Vec<u32>,
    thr: Vec<f32>,
    flags: Vec<u8>,
    left: Vec<i32>,
    right: Vec<i32>,
    leaves: Vec<f64>,
    roots: Vec<i32>,
    weights: Vec<f64>,
}

impl FlatForest {
    fn from_trees(trees: &[Tree]) -> Result<Self> {
        let mut flat = FlatForest::default();
        for tree in trees {
            let root = flat.flatten(&tree.root)?;
            flat.roots.push(root);
            flat.weights.push(tree.weight);
        }
        anyhow::ensure!(
            flat.feat.len() < i32::MAX as usize && flat.leaves.len() < i32::MAX as usize,
            "Forest too large for array layout ({} nodes, {} leaves)",
            flat.feat.len(),
            flat.leaves.len()
        );
        Ok(flat)
    }

    fn flatten(&mut self, node: &Node) -> Result<i32> {
        match node {
            Node::Leaf { value } => {
                self.leaves.push(*value);
                Ok(!((self.leaves.len() - 1) as i32))
            }
            Node::Split {
                feature_idx,
                split,
                left_child,
                right_child,
                missing_direction,
            } => {
                let (thr, flags) = encode_split(split, *missing_direction)?;
                let idx = self.feat.len();
                self.feat.push(*feature_idx as u32);
                self.thr.push(thr);
                self.flags.push(flags);
                self.left.push(0);
                self.right.push(0);
                let l = self.flatten(left_child)?;
                let r = self.flatten(right_child)?;
                self.left[idx] = l;
                self.right[idx] = r;
                Ok(idx as i32)
            }
        }
    }
}

fn encode_split(split: &SplitKind, missing_direction: MissingDirection) -> Result<(f32, u8)> {
    use crate::ir::Operator::*;
    let SplitKind::Numeric {
        threshold,
        operator,
    } = split
    else {
        bail!("Array layout supports numeric splits only");
    };

    let mut flags = match operator {
        LessThan => OP_LT,
        LessOrEqual => OP_LE,
        GreaterThan => OP_GT,
        GreaterOrEqual => OP_GE,
        Equal => OP_EQ,
        NotEqual => OP_NE,
    };
    match missing_direction {
        MissingDirection::Left => flags |= FLAG_MISSING_LEFT,
        MissingDirection::NaVsRest => flags |= FLAG_NA_VS_REST,
        // None and Right both route NaN right, the flag default.
        MissingDirection::Right | MissingDirection::None => {}
    }

    // NaVsRest never reads the threshold; MOJO stores a NaN sentinel there.
    // Store 0.0 so the THR array contains only ordinary literals.
    let thr = if threshold.is_nan() { 0.0 } else { *threshold };
    Ok((thr, flags))
}

/// Emit the static arrays and the loop-based tree walker shared by all paths.
fn build_data_and_walker(flat: &FlatForest) -> TokenStream {
    let n_nodes = flat.feat.len();
    let n_leaves = flat.leaves.len();
    let n_trees = flat.roots.len();

    let feat = &flat.feat;
    let flags = &flat.flags;
    let left = &flat.left;
    let right = &flat.right;
    let roots = &flat.roots;
    let weights = &flat.weights;
    let thr: Vec<TokenStream> = flat.thr.iter().map(|v| f32_token(*v)).collect();
    let leaves: Vec<TokenStream> = flat.leaves.iter().map(|v| f64_token(*v)).collect();

    let miss_left = FLAG_MISSING_LEFT;
    let na_vs_rest = FLAG_NA_VS_REST;
    let op_mask = OP_MASK;

    quote! {
        static FEAT: [u32; #n_nodes] = [#(#feat),*];
        static THR: [f32; #n_nodes] = [#(#thr),*];
        static FLAGS: [u8; #n_nodes] = [#(#flags),*];
        static LEFT: [i32; #n_nodes] = [#(#left),*];
        static RIGHT: [i32; #n_nodes] = [#(#right),*];
        static LEAVES: [f64; #n_leaves] = [#(#leaves),*];
        static ROOTS: [i32; #n_trees] = [#(#roots),*];
        static WEIGHTS: [f64; #n_trees] = [#(#weights),*];

        #[inline(always)]
        fn eval_tree(mut idx: i32, features: &[f32]) -> f64 {
            loop {
                if idx < 0 {
                    return LEAVES[(!idx) as usize];
                }
                let i = idx as usize;
                let val = features[FEAT[i] as usize];
                let flags = FLAGS[i];
                let go_left = if val.is_nan() {
                    (flags & #miss_left) != 0
                } else if (flags & #na_vs_rest) != 0 {
                    true
                } else {
                    let thr = THR[i];
                    match flags & #op_mask {
                        0 => val < thr,
                        1 => val <= thr,
                        2 => val > thr,
                        3 => val >= thr,
                        4 => val == thr,
                        _ => val != thr,
                    }
                };
                idx = if go_left { LEFT[i] } else { RIGHT[i] };
            }
        }
    }
}

/// Generate a standalone Rust source file using the array layout.
pub fn generate(forest: &Forest, no_std: bool) -> Result<String> {
    let flat = FlatForest::from_trees(&forest.trees)?;
    let data_and_walker = build_data_and_walker(&flat);
    let n_trees = flat.roots.len();

    if let PostTransform::Softmax { n_classes } = &forest.post_transform {
        let n_classes = *n_classes;
        anyhow::ensure!(
            n_classes >= 2,
            "Softmax requires at least 2 classes, got {}",
            n_classes
        );
        anyhow::ensure!(
            n_trees.is_multiple_of(n_classes),
            "Tree count ({}) must be divisible by n_classes ({}) for softmax",
            n_trees,
            n_classes
        );
        return Ok(build_softmax(forest, &data_and_walker, n_trees, n_classes, no_std).to_string());
    }

    // Accumulation mirrors the if/else layout exactly (same order, same f64
    // arithmetic) so both layouts produce bit-identical predictions.
    let raw_decl = match forest.aggregation {
        AggregationKind::Sum => {
            let base = forest.base_score;
            quote! {
                let mut acc: f64 = #base;
                for t in 0..#n_trees {
                    acc += eval_tree(ROOTS[t], features) * WEIGHTS[t];
                }
                let raw = acc;
            }
        }
        AggregationKind::Average => {
            let n = n_trees as f64;
            quote! {
                let mut acc: f64 = 0.0;
                for t in 0..#n_trees {
                    acc += eval_tree(ROOTS[t], features) * WEIGHTS[t];
                }
                let raw = acc / #n;
            }
        }
    };
    let post = post_transform_expr(&forest.post_transform, no_std);

    let output = quote! {
        /// Auto-generated by bonsai (array layout).
        /// Do not edit manually.
        pub struct Model;

        #[allow(dead_code)]
        impl Model {
            pub const N_CLASSES: usize = 1;

            pub fn predict(&self, features: &[f32]) -> f32 {
                #raw_decl
                #post
            }

            pub fn predict_batch(&self, features: &[f32], n_features: usize, out: &mut [f32]) {
                for (i, row) in features.chunks_exact(n_features).enumerate() {
                    out[i] = self.predict(row);
                }
            }
        }

        #data_and_walker
    };
    Ok(output.to_string())
}

fn build_softmax(
    forest: &Forest,
    data_and_walker: &TokenStream,
    n_trees: usize,
    n_classes: usize,
    no_std: bool,
) -> TokenStream {
    let bases: Vec<f64> = (0..n_classes)
        .map(|c| {
            if c < forest.base_scores.len() {
                forest.base_scores[c]
            } else {
                forest.base_score
            }
        })
        .collect();

    // Match the if/else softmax: Sum starts from the per-class base, Average
    // starts from zero and divides by the per-class tree count.
    let (init, avg_div) = match forest.aggregation {
        AggregationKind::Sum => (quote! { [#(#bases),*] }, quote! {}),
        AggregationKind::Average => {
            let per_class = (n_trees / n_classes) as f64;
            (
                quote! { [0.0f64; #n_classes] },
                quote! {
                    for r in raw.iter_mut() {
                        *r /= #per_class;
                    }
                },
            )
        }
    };

    let exp = exp_expr(quote! { raw[j] - max_raw }, no_std);
    let core_compute = quote! {
        let mut raw: [f64; #n_classes] = #init;
        for t in 0..#n_trees {
            raw[t % #n_classes] += eval_tree(ROOTS[t], features) * WEIGHTS[t];
        }
        #avg_div

        let mut max_raw = raw[0];
        for r in raw.iter().skip(1) {
            if *r > max_raw {
                max_raw = *r;
            }
        }
        let mut e = [0.0f64; #n_classes];
        let mut sum_e = 0.0f64;
        for j in 0..#n_classes {
            e[j] = #exp;
            sum_e += e[j];
        }
    };

    let methods = if no_std {
        quote! {
            pub fn predict_proba_into(&self, features: &[f32], out: &mut [f32]) {
                #core_compute
                for j in 0..#n_classes {
                    out[j] = (e[j] / sum_e) as f32;
                }
            }

            pub fn predict_batch(&self, features: &[f32], n_features: usize, out: &mut [f32]) {
                let n_classes = #n_classes;
                for (i, row) in features.chunks_exact(n_features).enumerate() {
                    self.predict_proba_into(row, &mut out[i * n_classes..(i + 1) * n_classes]);
                }
            }
        }
    } else {
        quote! {
            pub fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
                #core_compute
                let mut out = Vec::with_capacity(#n_classes);
                for j in 0..#n_classes {
                    out.push((e[j] / sum_e) as f32);
                }
                out
            }

            pub fn predict_batch(&self, features: &[f32], n_features: usize, out: &mut [f32]) {
                let n_classes = #n_classes;
                for (i, row) in features.chunks_exact(n_features).enumerate() {
                    let probs = self.predict_proba(row);
                    for (j, prob) in probs.into_iter().enumerate() {
                        out[i * n_classes + j] = prob;
                    }
                }
            }
        }
    };

    quote! {
        /// Auto-generated by bonsai (array layout).
        /// Do not edit manually.
        pub struct Model;

        #[allow(dead_code)]
        impl Model {
            pub const N_CLASSES: usize = #n_classes;

            #methods
        }

        #data_and_walker
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backends::rust::{
        generate_with_layout, generate_with_options, resolve_layout, CodegenOptions, Layout,
        ARRAY_LAYOUT_NODE_THRESHOLD,
    };
    use crate::ir::{AggregationKind, MissingDirection, Node, Operator, PostTransform, Tree};
    use crate::testutil::compile_and_run;

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

    fn forest(trees: Vec<Tree>, agg: AggregationKind, pt: PostTransform, base: f64) -> Forest {
        Forest {
            trees,
            aggregation: agg,
            post_transform: pt,
            base_score: base,
            base_scores: vec![],
            catboost_metadata: None,
        }
    }

    #[test]
    fn test_flatten_encoding() {
        let tree = Tree {
            root: split(
                3,
                0.5,
                Operator::LessThan,
                MissingDirection::Left,
                leaf(1.0),
                leaf(2.0),
            ),
            weight: 2.0,
        };
        let flat = FlatForest::from_trees(&[tree]).unwrap();
        assert_eq!(flat.feat, vec![3]);
        assert_eq!(flat.thr, vec![0.5]);
        assert_eq!(flat.flags, vec![OP_LT | FLAG_MISSING_LEFT]);
        assert_eq!(flat.left, vec![!0]);
        assert_eq!(flat.right, vec![!1]);
        assert_eq!(flat.leaves, vec![1.0, 2.0]);
        assert_eq!(flat.roots, vec![0]);
        assert_eq!(flat.weights, vec![2.0]);
    }

    #[test]
    fn test_navsrest_threshold_sanitized() {
        let (thr, flags) = encode_split(
            &SplitKind::Numeric {
                threshold: f32::NAN,
                operator: Operator::LessThan,
            },
            MissingDirection::NaVsRest,
        )
        .unwrap();
        assert_eq!(thr, 0.0);
        assert_eq!(flags, OP_LT | FLAG_NA_VS_REST);
    }

    #[test]
    fn test_categorical_split_rejected() {
        let err = encode_split(
            &SplitKind::Categorical {
                bitoff: 0,
                nbits: 8,
                data: vec![0xFF],
            },
            MissingDirection::None,
        )
        .unwrap_err();
        assert!(err.to_string().contains("numeric splits only"));
    }

    #[test]
    fn test_leaf_only_tree_root() {
        let flat = FlatForest::from_trees(&[Tree {
            root: leaf(4.5),
            weight: 1.0,
        }])
        .unwrap();
        assert_eq!(flat.roots, vec![!0]);
        assert_eq!(flat.leaves, vec![4.5]);
    }

    fn asymmetric_tree() -> Tree {
        Tree {
            root: split(
                0,
                0.5,
                Operator::LessThan,
                MissingDirection::Right,
                split(
                    1,
                    0.3,
                    Operator::LessOrEqual,
                    MissingDirection::Left,
                    leaf(1.0),
                    leaf(2.0),
                ),
                leaf(3.0),
            ),
            weight: 1.0,
        }
    }

    #[test]
    fn test_resolve_layout_auto_small_stays_ifelse() {
        let f = forest(
            vec![asymmetric_tree()],
            AggregationKind::Sum,
            PostTransform::Identity,
            0.0,
        );
        assert_eq!(resolve_layout(&f, Layout::Auto).unwrap(), Layout::IfElse);
    }

    #[test]
    fn test_resolve_layout_auto_large_switches_to_array() {
        // 5 nodes per asymmetric tree; exceed the threshold.
        let n_trees = ARRAY_LAYOUT_NODE_THRESHOLD / 5 + 1;
        let trees = (0..n_trees).map(|_| asymmetric_tree()).collect();
        let f = forest(trees, AggregationKind::Sum, PostTransform::Identity, 0.0);
        assert_eq!(resolve_layout(&f, Layout::Auto).unwrap(), Layout::Array);
    }

    #[test]
    fn test_resolve_layout_array_rejects_categorical() {
        let tree = Tree {
            root: Node::Split {
                feature_idx: 0,
                split: SplitKind::Categorical {
                    bitoff: 0,
                    nbits: 8,
                    data: vec![0x01],
                },
                left_child: Box::new(leaf(1.0)),
                right_child: Box::new(leaf(2.0)),
                missing_direction: MissingDirection::None,
            },
            weight: 1.0,
        };
        let f = forest(
            vec![tree],
            AggregationKind::Sum,
            PostTransform::Identity,
            0.0,
        );
        assert!(resolve_layout(&f, Layout::Array).is_err());
        // Auto quietly falls back instead.
        assert_eq!(resolve_layout(&f, Layout::Auto).unwrap(), Layout::IfElse);
    }

    #[test]
    fn test_generated_structure() {
        let f = forest(
            vec![asymmetric_tree()],
            AggregationKind::Sum,
            PostTransform::Logit,
            0.5,
        );
        let code = generate(&f, false).unwrap();
        assert!(code.contains("static FEAT"));
        assert!(code.contains("static LEAVES"));
        assert!(code.contains("fn eval_tree"));
        assert!(code.contains("pub struct Model"));
        assert!(!code.contains("fn tree_0"));
    }

    // -----------------------------------------------------------------------
    // Differential tests: compile both layouts with rustc and require
    // bit-identical predictions.
    // -----------------------------------------------------------------------

    const SCALAR_DRIVER: &str = r#"
mod model { include!("model.rs"); }
fn main() {
    let rows: &[[f32; 3]] = &[
        [0.1, 5.0, -2.0],
        [0.9, -1.5, 3.5],
        [f32::NAN, 0.0, 1.0],
        [0.5, f32::NAN, f32::NAN],
        [-3.0, 2.5, 0.49],
        [1e30, -1e30, 0.0],
        [0.5, 0.3, 0.0],
    ];
    let m = model::Model;
    for r in rows {
        println!("{:08x}", m.predict(r.as_slice()).to_bits());
    }
}
"#;

    const PROBA_DRIVER: &str = r#"
mod model { include!("model.rs"); }
fn main() {
    let rows: &[[f32; 3]] = &[
        [0.1, 5.0, -2.0],
        [0.9, -1.5, 3.5],
        [f32::NAN, 0.0, 1.0],
        [0.5, f32::NAN, f32::NAN],
    ];
    let m = model::Model;
    for r in rows {
        for p in m.predict_proba(r.as_slice()) {
            println!("{:08x}", p.to_bits());
        }
    }
}
"#;

    /// A forest exercising every operator, all four missing directions,
    /// non-unit weights, and an NaVsRest split.
    fn differential_trees() -> Vec<Tree> {
        vec![
            asymmetric_tree(),
            Tree {
                // NaVsRest: threshold is an unused NaN sentinel, as in MOJO.
                root: split(
                    1,
                    f32::NAN,
                    Operator::LessThan,
                    MissingDirection::NaVsRest,
                    leaf(0.25),
                    leaf(-0.25),
                ),
                weight: 0.5,
            },
            Tree {
                root: split(
                    0,
                    0.0,
                    Operator::GreaterThan,
                    MissingDirection::None,
                    split(
                        1,
                        -1.5,
                        Operator::Equal,
                        MissingDirection::Right,
                        leaf(5.0),
                        split(
                            2,
                            0.0,
                            Operator::NotEqual,
                            MissingDirection::Left,
                            leaf(-5.0),
                            leaf(0.5),
                        ),
                    ),
                    split(
                        2,
                        1.0,
                        Operator::GreaterOrEqual,
                        MissingDirection::Right,
                        leaf(0.1),
                        leaf(-0.1),
                    ),
                ),
                weight: 2.0,
            },
        ]
    }

    #[test]
    fn test_differential_scalar_layouts_match() {
        for agg in [AggregationKind::Sum, AggregationKind::Average] {
            let f = forest(differential_trees(), agg.clone(), PostTransform::Logit, 0.5);
            let ifelse = generate_with_layout(&f, Layout::IfElse).unwrap();
            let array = generate_with_layout(&f, Layout::Array).unwrap();
            let out_ifelse = compile_and_run(&ifelse, SCALAR_DRIVER);
            let out_array = compile_and_run(&array, SCALAR_DRIVER);
            assert_eq!(out_ifelse, out_array, "layouts diverged for {:?}", agg);
        }
    }

    #[test]
    fn test_differential_softmax_layouts_match() {
        // 6 trees, 3 classes, round-robin assignment.
        let mut trees = differential_trees();
        trees.extend(differential_trees());
        let mut f = forest(
            trees,
            AggregationKind::Sum,
            PostTransform::Softmax { n_classes: 3 },
            0.0,
        );
        f.base_scores = vec![0.1, -0.2, 0.3];
        let ifelse = generate_with_layout(&f, Layout::IfElse).unwrap();
        let array = generate_with_layout(&f, Layout::Array).unwrap();
        assert_eq!(
            compile_and_run(&ifelse, PROBA_DRIVER),
            compile_and_run(&array, PROBA_DRIVER)
        );
    }

    // -----------------------------------------------------------------------
    // no_std mode
    // -----------------------------------------------------------------------

    fn no_std_opts(layout: Layout) -> CodegenOptions {
        CodegenOptions {
            layout,
            no_std: true,
        }
    }

    /// Compile `model_code` inside a `#![no_std]` library crate; fails if the
    /// generated code touches anything outside core.
    fn compile_no_std_lib(model_code: &str) {
        use std::process::Command;
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("model.rs"), model_code).unwrap();
        std::fs::write(
            dir.path().join("lib.rs"),
            "#![no_std]\n#[allow(dead_code)]\nmod model { include!(\"model.rs\"); }\n",
        )
        .unwrap();
        let out = Command::new("rustc")
            .arg("--edition")
            .arg("2021")
            .arg("--crate-type")
            .arg("lib")
            .arg("--out-dir")
            .arg(dir.path())
            .arg(dir.path().join("lib.rs"))
            .output()
            .expect("failed to invoke rustc");
        assert!(
            out.status.success(),
            "no_std compile failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    #[test]
    fn test_no_std_identity_compiles_core_only() {
        let f = forest(
            differential_trees(),
            AggregationKind::Sum,
            PostTransform::Identity,
            0.5,
        );
        for layout in [Layout::IfElse, Layout::Array] {
            let code = generate_with_options(&f, no_std_opts(layout)).unwrap();
            compile_no_std_lib(&code);
        }
    }

    #[test]
    fn test_no_std_identity_matches_std_output() {
        let f = forest(
            differential_trees(),
            AggregationKind::Sum,
            PostTransform::Identity,
            0.5,
        );
        for layout in [Layout::IfElse, Layout::Array] {
            let std_code = generate_with_layout(&f, layout).unwrap();
            let no_std_code = generate_with_options(&f, no_std_opts(layout)).unwrap();
            assert_eq!(
                compile_and_run(&std_code, SCALAR_DRIVER),
                compile_and_run(&no_std_code, SCALAR_DRIVER),
                "no_std output diverged for {:?}",
                layout
            );
        }
    }

    #[test]
    fn test_no_std_logit_uses_libm() {
        let f = forest(
            differential_trees(),
            AggregationKind::Sum,
            PostTransform::Logit,
            0.0,
        );
        for layout in [Layout::IfElse, Layout::Array] {
            let code = generate_with_options(&f, no_std_opts(layout)).unwrap();
            assert!(code.contains("libm"), "expected libm exp for {:?}", layout);
        }
    }

    #[test]
    fn test_no_std_softmax_uses_into_api() {
        let mut trees = differential_trees();
        trees.extend(differential_trees());
        let f = forest(
            trees,
            AggregationKind::Sum,
            PostTransform::Softmax { n_classes: 3 },
            0.0,
        );
        for layout in [Layout::IfElse, Layout::Array] {
            let code = generate_with_options(&f, no_std_opts(layout)).unwrap();
            assert!(code.contains("predict_proba_into"));
            assert!(!code.contains("Vec"), "no_std code must not allocate");
        }
    }

    #[test]
    fn test_no_std_rejects_catboost_ctr() {
        let mut f = forest(
            differential_trees(),
            AggregationKind::Sum,
            PostTransform::Identity,
            0.0,
        );
        f.catboost_metadata = Some(crate::ir::CatboostMetadata {
            ctrs: vec![],
            ctr_data: std::collections::HashMap::new(),
        });
        let err = generate_with_options(&f, no_std_opts(Layout::IfElse))
            .unwrap_err()
            .to_string();
        assert!(err.contains("no-std"), "unexpected error: {err}");
    }

    #[test]
    fn test_differential_oblivious_vs_array() {
        // A depth-2 oblivious tree (same split per level). The if/else layout
        // compiles this through the oblivious fast path; the array layout uses
        // the generic walker. Both must agree with plain if/else semantics.
        let level1 =
            |l: Node, r: Node| split(1, 0.3, Operator::LessThan, MissingDirection::Right, l, r);
        let tree = Tree {
            root: split(
                0,
                0.5,
                Operator::LessThan,
                MissingDirection::Right,
                level1(leaf(1.0), leaf(2.0)),
                level1(leaf(3.0), leaf(4.0)),
            ),
            weight: 1.0,
        };
        let f = forest(
            vec![tree],
            AggregationKind::Sum,
            PostTransform::Identity,
            0.0,
        );
        let ifelse = generate_with_layout(&f, Layout::IfElse).unwrap();
        assert!(
            ifelse.contains("const LEAVES"),
            "expected oblivious fast path in if/else layout"
        );
        let array = generate_with_layout(&f, Layout::Array).unwrap();
        assert_eq!(
            compile_and_run(&ifelse, SCALAR_DRIVER),
            compile_and_run(&array, SCALAR_DRIVER)
        );
    }
}
