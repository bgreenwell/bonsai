/// Comparison operator for numeric splits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Operator {
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    Equal,
    NotEqual,
}

/// Where a NaN value routes at a split node.
///
/// Left / Right: NaN goes to that child; non-NaN follows the comparison.
/// NaVsRest: Split solely on NaN-ness (non-NaN→left, NaN→right). The threshold is ignored.
///   Preserved as a distinct variant for source-format round-trip debuggability.
/// None: format did not specify; backend defaults to Right.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MissingDirection {
    Left,
    Right,
    NaVsRest,
    None,
}

/// The split test at an internal node.
#[derive(Debug, Clone)]
pub enum SplitKind {
    /// Standard threshold comparison: `feature[i] <op> threshold`.
    Numeric { threshold: f32, operator: Operator },
    /// Categorical membership via bitset.
    /// Condition is TRUE when the integer category value is a member of the set.
    Categorical {
        bitoff: u16,
        nbits: u32,
        data: Vec<u8>,
    },
    /// CatBoost-specific categorical split via Online CTR (Counter).
    /// The actual split logic is handled by the backend using embedded CTR tables.
    OnlineCtr { ctr_idx: usize, threshold: f32 },
}

#[derive(Debug, Clone)]
pub struct CatboostMetadata {
    pub ctrs: Vec<CtrInfo>,
    pub ctr_data: std::collections::HashMap<String, CtrValueTable>,
}

#[derive(Debug, Clone)]
pub struct CtrInfo {
    pub elements: Vec<CtrElement>,
    pub identifier: String,
    pub prior_numerator: f64,
    pub prior_denominator: f64,
    pub scale: f64,
    pub shift: f64,
}

#[derive(Debug, Clone)]
pub struct CtrElement {
    pub cat_feature_index: usize,
    pub combination_element: String,
}

#[derive(Debug, Clone)]
pub struct CtrValueTable {
    pub hash_map: Vec<u64>, // [hash, val1, val2, ...] flattened
    pub hash_stride: usize,
    pub counter_denominator: i64,
}

/// A node in the decision tree.
///
/// Convention: `left_child` is taken when the split condition evaluates to TRUE.
/// For Numeric: condition = `feature[feature_idx] <op> threshold`.
/// For Categorical: condition = `!bitset_contains(feature[feature_idx] as i32)`.
///   **CRITICAL**: Category value NOT in bitset → LEFT (TRUE). Category IN bitset → RIGHT.
///   This matches H2O MOJO semantics where membership tests follow RIGHT child.
#[derive(Debug, Clone)]
pub enum Node {
    Split {
        feature_idx: usize,
        split: SplitKind,
        left_child: Box<Node>,
        right_child: Box<Node>,
        missing_direction: MissingDirection,
    },
    Leaf {
        value: f64,
    },
}

/// A single decision tree in the ensemble.
#[derive(Debug, Clone)]
pub struct Tree {
    pub root: Node,
    /// Per-tree weight. 1.0 for standard GBM / RF / DRF.
    pub weight: f64,
}

impl Node {
    /// Walk the node tree to check if it's perfectly oblivious (symmetric).
    /// If it is, returns the sequence of split parameters from root to leaf.
    pub fn get_oblivious_splits(&self) -> Option<Vec<(usize, SplitKind, MissingDirection)>> {
        let mut splits = Vec::new();
        if self.collect_oblivious_splits(0, &mut splits) && self.is_symmetric(0, splits.len()) {
            Some(splits)
        } else {
            None
        }
    }

    fn collect_oblivious_splits(
        &self,
        depth: usize,
        splits: &mut Vec<(usize, SplitKind, MissingDirection)>,
    ) -> bool {
        match self {
            Node::Leaf { .. } => true,
            Node::Split {
                feature_idx,
                split,
                left_child,
                right_child,
                missing_direction,
            } => {
                if splits.len() <= depth {
                    splits.push((*feature_idx, split.clone(), *missing_direction));
                } else {
                    let (f, s, m) = &splits[depth];
                    if f != feature_idx || !split_kind_eq(s, split) || m != missing_direction {
                        return false;
                    }
                }
                left_child.collect_oblivious_splits(depth + 1, splits)
                    && right_child.collect_oblivious_splits(depth + 1, splits)
            }
        }
    }

    fn is_symmetric(&self, depth: usize, max_depth: usize) -> bool {
        match self {
            Node::Leaf { .. } => depth == max_depth,
            Node::Split {
                left_child,
                right_child,
                ..
            } => {
                depth < max_depth
                    && left_child.is_symmetric(depth + 1, max_depth)
                    && right_child.is_symmetric(depth + 1, max_depth)
            }
        }
    }

    /// Walk the node tree to collect all leaf values in RIGHT-to-LEFT order.
    /// This matches the oblivious index bit logic (0=right, 1=left).
    pub fn collect_leaves(&self, out: &mut Vec<f64>) {
        match self {
            Node::Leaf { value } => out.push(*value),
            Node::Split {
                left_child,
                right_child,
                ..
            } => {
                right_child.collect_leaves(out);
                left_child.collect_leaves(out);
            }
        }
    }
}

fn split_kind_eq(a: &SplitKind, b: &SplitKind) -> bool {
    match (a, b) {
        (
            SplitKind::Numeric {
                threshold: t1,
                operator: o1,
            },
            SplitKind::Numeric {
                threshold: t2,
                operator: o2,
            },
        ) => t1 == t2 && o1 == o2,
        (
            SplitKind::Categorical {
                bitoff: o1,
                nbits: n1,
                data: d1,
            },
            SplitKind::Categorical {
                bitoff: o2,
                nbits: n2,
                data: d2,
            },
        ) => o1 == o2 && n1 == n2 && d1 == d2,
        (
            SplitKind::OnlineCtr {
                ctr_idx: i1,
                threshold: t1,
            },
            SplitKind::OnlineCtr {
                ctr_idx: i2,
                threshold: t2,
            },
        ) => i1 == i2 && t1 == t2,
        _ => false,
    }
}

/// How tree outputs are combined before the post-transform is applied.
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationKind {
    /// GBM: `result = base_score + Σ(tree_i × weight_i)`
    Sum,
    /// DRF / RF: `result = Σ(tree_i) / n_trees`  (base_score is NOT added)
    Average,
}

/// Post-transform applied to the aggregated scalar (or vector for multiclass).
#[derive(Debug, Clone, PartialEq)]
pub enum PostTransform {
    /// `output = raw`
    Identity,
    /// `output = 1 / (1 + exp(-raw))`  (logistic sigmoid)
    Logit,
    /// `output = exp(raw)`
    Log,
    /// Multiclass softmax over K raw scores.
    ///
    /// Trees are assigned to classes round-robin: tree[i] → class (i % n_classes).
    /// Output is a Vec<f32> of length `n_classes`.
    Softmax { n_classes: usize },
}

/// The complete ensemble model.
/// Single artifact: a Frontend produces one, a Backend consumes one.
#[derive(Debug, Clone)]
pub struct Forest {
    pub trees: Vec<Tree>,
    /// Scalar bias added after aggregation (regression / binary classification).
    /// f64 to preserve precision from H2O's init_f.
    pub base_score: f64,
    /// Per-class biases for multiclass softmax (XGBoost 3.x stores one value per class).
    /// When non-empty, `base_scores[c]` replaces `base_score` for class c.
    /// Length must equal `n_classes` for Softmax models; empty for all other models.
    pub base_scores: Vec<f64>,
    pub aggregation: AggregationKind,
    pub post_transform: PostTransform,
    /// Metadata for CatBoost models using categorical features.
    pub catboost_metadata: Option<CatboostMetadata>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oblivious_tree_detection() {
        let split_kind = SplitKind::Numeric {
            threshold: 0.5,
            operator: Operator::GreaterThan,
        };
        // Depth 2 oblivious tree
        let root = Node::Split {
            feature_idx: 1,
            split: split_kind.clone(),
            left_child: Box::new(Node::Split {
                feature_idx: 0,
                split: split_kind.clone(),
                left_child: Box::new(Node::Leaf { value: 3.0 }),
                right_child: Box::new(Node::Leaf { value: 2.0 }),
                missing_direction: MissingDirection::None,
            }),
            right_child: Box::new(Node::Split {
                feature_idx: 0,
                split: split_kind.clone(),
                left_child: Box::new(Node::Leaf { value: 1.0 }),
                right_child: Box::new(Node::Leaf { value: 0.0 }),
                missing_direction: MissingDirection::None,
            }),
            missing_direction: MissingDirection::None,
        };

        let splits = root.get_oblivious_splits().expect("Should be oblivious");
        assert_eq!(splits.len(), 2);
        assert_eq!(splits[0].0, 1); // depth 0: root
        assert_eq!(splits[1].0, 0); // depth 1: children

        let mut leaves = Vec::new();
        root.collect_leaves(&mut leaves);
        assert_eq!(leaves, vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_simple_leaf_tree() {
        let tree = Tree {
            root: Node::Leaf { value: 0.42 },
            weight: 1.0,
        };

        match tree.root {
            Node::Leaf { value } => assert_eq!(value, 0.42),
            _ => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_numeric_split_node() {
        let tree = Tree {
            root: Node::Split {
                feature_idx: 0,
                split: SplitKind::Numeric {
                    threshold: 5.0,
                    operator: Operator::LessThan,
                },
                left_child: Box::new(Node::Leaf { value: 1.0 }),
                right_child: Box::new(Node::Leaf { value: -1.0 }),
                missing_direction: MissingDirection::Right,
            },
            weight: 1.0,
        };

        match tree.root {
            Node::Split {
                feature_idx,
                split,
                missing_direction,
                ..
            } => {
                assert_eq!(feature_idx, 0);
                assert_eq!(missing_direction, MissingDirection::Right);
                match split {
                    SplitKind::Numeric {
                        threshold,
                        operator,
                    } => {
                        assert_eq!(threshold, 5.0);
                        assert!(matches!(operator, Operator::LessThan));
                    }
                    _ => panic!("Expected numeric split"),
                }
            }
            _ => panic!("Expected split node"),
        }
    }

    #[test]
    fn test_categorical_split_node() {
        // Test bitset encoding: bits 0,1,3 set (representing categories 0,1,3)
        // Binary: 00001011 = 0x0B
        let bitset_data = vec![0x0B];

        let tree = Tree {
            root: Node::Split {
                feature_idx: 2,
                split: SplitKind::Categorical {
                    bitoff: 0,
                    nbits: 4,
                    data: bitset_data.clone(),
                },
                left_child: Box::new(Node::Leaf { value: 10.0 }),
                right_child: Box::new(Node::Leaf { value: -10.0 }),
                missing_direction: MissingDirection::Right,
            },
            weight: 1.0,
        };

        match tree.root {
            Node::Split {
                feature_idx, split, ..
            } => {
                assert_eq!(feature_idx, 2);
                match split {
                    SplitKind::Categorical {
                        bitoff,
                        nbits,
                        data,
                    } => {
                        assert_eq!(bitoff, 0);
                        assert_eq!(nbits, 4);
                        assert_eq!(data, bitset_data);
                    }
                    _ => panic!("Expected categorical split"),
                }
            }
            _ => panic!("Expected split node"),
        }
    }

    #[test]
    fn test_missing_direction_variants() {
        let directions = [
            MissingDirection::Left,
            MissingDirection::Right,
            MissingDirection::NaVsRest,
            MissingDirection::None,
        ];

        // Test that all variants are distinct
        for (i, dir1) in directions.iter().enumerate() {
            for (j, dir2) in directions.iter().enumerate() {
                if i == j {
                    assert_eq!(dir1, dir2);
                } else {
                    assert_ne!(dir1, dir2);
                }
            }
        }
    }

    #[test]
    fn test_forest_regression_identity() {
        let forest = Forest {
            trees: vec![Tree {
                root: Node::Leaf { value: 1.5 },
                weight: 1.0,
            }],
            base_score: 0.5,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };

        assert_eq!(forest.trees.len(), 1);
        assert_eq!(forest.base_score, 0.5);
        assert_eq!(forest.aggregation, AggregationKind::Sum);
        assert_eq!(forest.post_transform, PostTransform::Identity);
    }

    #[test]
    fn test_forest_classification_logit() {
        let forest = Forest {
            trees: vec![
                Tree {
                    root: Node::Leaf { value: 0.1 },
                    weight: 1.0,
                },
                Tree {
                    root: Node::Leaf { value: -0.1 },
                    weight: 1.0,
                },
            ],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Logit,
            catboost_metadata: None,
        };

        assert_eq!(forest.trees.len(), 2);
        assert_eq!(forest.post_transform, PostTransform::Logit);
    }

    #[test]
    fn test_empty_forest() {
        let forest = Forest {
            trees: vec![],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };

        assert_eq!(forest.trees.len(), 0);
    }

    #[test]
    fn test_operator_variants() {
        let operators = [
            Operator::LessThan,
            Operator::LessOrEqual,
            Operator::GreaterThan,
            Operator::GreaterOrEqual,
            Operator::Equal,
            Operator::NotEqual,
        ];

        // Verify all operators are distinct
        for (i, op1) in operators.iter().enumerate() {
            for (j, op2) in operators.iter().enumerate() {
                if i == j {
                    assert_eq!(op1, op2);
                } else {
                    assert_ne!(op1, op2);
                }
            }
        }
    }

    #[test]
    fn test_deep_tree_structure() {
        // Create a tree with depth 3
        let tree = Tree {
            root: Node::Split {
                feature_idx: 0,
                split: SplitKind::Numeric {
                    threshold: 10.0,
                    operator: Operator::LessThan,
                },
                left_child: Box::new(Node::Split {
                    feature_idx: 1,
                    split: SplitKind::Numeric {
                        threshold: 5.0,
                        operator: Operator::LessThan,
                    },
                    left_child: Box::new(Node::Leaf { value: 1.0 }),
                    right_child: Box::new(Node::Leaf { value: 2.0 }),
                    missing_direction: MissingDirection::Left,
                }),
                right_child: Box::new(Node::Split {
                    feature_idx: 2,
                    split: SplitKind::Numeric {
                        threshold: 15.0,
                        operator: Operator::GreaterThan,
                    },
                    left_child: Box::new(Node::Leaf { value: 3.0 }),
                    right_child: Box::new(Node::Leaf { value: 4.0 }),
                    missing_direction: MissingDirection::Right,
                }),
                missing_direction: MissingDirection::None,
            },
            weight: 1.0,
        };

        // Verify we can navigate the tree structure
        match &tree.root {
            Node::Split {
                left_child,
                right_child,
                ..
            } => {
                assert!(matches!(**left_child, Node::Split { .. }));
                assert!(matches!(**right_child, Node::Split { .. }));
            }
            _ => panic!("Expected split at root"),
        }
    }

    #[test]
    fn test_categorical_bitset_multiple_bytes() {
        // Test bitset with multiple bytes
        // Represents 16 categories with various bits set
        let bitset_data = vec![0xAA, 0x55]; // Alternating bit pattern

        let split = SplitKind::Categorical {
            bitoff: 0,
            nbits: 16,
            data: bitset_data.clone(),
        };

        match split {
            SplitKind::Categorical {
                bitoff,
                nbits,
                data,
            } => {
                assert_eq!(bitoff, 0);
                assert_eq!(nbits, 16);
                assert_eq!(data.len(), 2);
                assert_eq!(data[0], 0xAA);
                assert_eq!(data[1], 0x55);
            }
            _ => panic!("Expected categorical split"),
        }
    }

    #[test]
    fn test_na_vs_rest_split() {
        // NaVsRest is a special case where split is solely on NaN-ness
        let tree = Tree {
            root: Node::Split {
                feature_idx: 5,
                split: SplitKind::Numeric {
                    threshold: f32::NAN,
                    operator: Operator::LessThan,
                },
                left_child: Box::new(Node::Leaf { value: 0.0 }),
                right_child: Box::new(Node::Leaf { value: 1.0 }),
                missing_direction: MissingDirection::NaVsRest,
            },
            weight: 1.0,
        };

        match tree.root {
            Node::Split {
                split,
                missing_direction,
                ..
            } => {
                assert_eq!(missing_direction, MissingDirection::NaVsRest);
                match split {
                    SplitKind::Numeric { threshold, .. } => {
                        assert!(threshold.is_nan());
                    }
                    _ => panic!("Expected numeric split"),
                }
            }
            _ => panic!("Expected split node"),
        }
    }

    #[test]
    fn test_tree_weights() {
        let trees = [
            Tree {
                root: Node::Leaf { value: 1.0 },
                weight: 0.5,
            },
            Tree {
                root: Node::Leaf { value: 2.0 },
                weight: 1.5,
            },
            Tree {
                root: Node::Leaf { value: 3.0 },
                weight: 2.0,
            },
        ];

        assert_eq!(trees[0].weight, 0.5);
        assert_eq!(trees[1].weight, 1.5);
        assert_eq!(trees[2].weight, 2.0);
    }

    #[test]
    fn test_aggregation_kinds() {
        let sum_forest = Forest {
            trees: vec![],
            base_score: 0.5,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };

        let avg_forest = Forest {
            trees: vec![],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Average,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };

        assert_eq!(sum_forest.aggregation, AggregationKind::Sum);
        assert_eq!(avg_forest.aggregation, AggregationKind::Average);
        assert_ne!(sum_forest.aggregation, avg_forest.aggregation);
    }

    #[test]
    fn test_post_transform_kinds() {
        let transforms = [
            PostTransform::Identity,
            PostTransform::Logit,
            PostTransform::Log,
        ];

        for (i, t1) in transforms.iter().enumerate() {
            for (j, t2) in transforms.iter().enumerate() {
                if i == j {
                    assert_eq!(t1, t2);
                } else {
                    assert_ne!(t1, t2);
                }
            }
        }
    }

    #[test]
    fn test_categorical_bitset_non_zero_offset() {
        // Test bitset with non-zero offset
        // This represents categories 5-8 (offset=5, 4 bits)
        let split = SplitKind::Categorical {
            bitoff: 5,
            nbits: 4,
            data: vec![0x0F], // All 4 bits set
        };

        match split {
            SplitKind::Categorical {
                bitoff,
                nbits,
                data,
            } => {
                assert_eq!(bitoff, 5);
                assert_eq!(nbits, 4);
                assert_eq!(data.len(), 1);
            }
            _ => panic!("Expected categorical split"),
        }
    }

    #[test]
    fn test_categorical_bitset_ragged_bits() {
        // Test bitset where nbits is not a multiple of 8
        // 10 bits requires 2 bytes (with 6 bits unused in second byte)
        let split = SplitKind::Categorical {
            bitoff: 0,
            nbits: 10,
            data: vec![0xFF, 0x03], // 8 bits + 2 bits = 10 bits total
        };

        match split {
            SplitKind::Categorical {
                bitoff: _,
                nbits,
                data,
            } => {
                assert_eq!(nbits, 10);
                assert_eq!(data.len(), 2);
                assert_eq!(data[0], 0xFF); // All 8 bits in first byte
                assert_eq!(data[1], 0x03); // Only lowest 2 bits in second byte
            }
            _ => panic!("Expected categorical split"),
        }
    }

    #[test]
    fn test_large_feature_indices() {
        // Test with very large feature indices to ensure no overflow
        let tree = Tree {
            root: Node::Split {
                feature_idx: 999_999,
                split: SplitKind::Numeric {
                    threshold: 0.5,
                    operator: Operator::LessThan,
                },
                left_child: Box::new(Node::Leaf { value: 1.0 }),
                right_child: Box::new(Node::Leaf { value: -1.0 }),
                missing_direction: MissingDirection::Right,
            },
            weight: 1.0,
        };

        match tree.root {
            Node::Split { feature_idx, .. } => {
                assert_eq!(feature_idx, 999_999);
            }
            _ => panic!("Expected split node"),
        }
    }

    #[test]
    fn test_deep_tree_leaf_traversal() {
        // Create a deeper tree and verify we can traverse to specific leaves
        let tree = Tree {
            root: Node::Split {
                feature_idx: 0,
                split: SplitKind::Numeric {
                    threshold: 10.0,
                    operator: Operator::LessThan,
                },
                left_child: Box::new(Node::Split {
                    feature_idx: 1,
                    split: SplitKind::Numeric {
                        threshold: 5.0,
                        operator: Operator::LessThan,
                    },
                    left_child: Box::new(Node::Split {
                        feature_idx: 2,
                        split: SplitKind::Numeric {
                            threshold: 2.0,
                            operator: Operator::LessThan,
                        },
                        left_child: Box::new(Node::Leaf { value: 1.0 }),
                        right_child: Box::new(Node::Leaf { value: 2.0 }),
                        missing_direction: MissingDirection::None,
                    }),
                    right_child: Box::new(Node::Leaf { value: 3.0 }),
                    missing_direction: MissingDirection::Left,
                }),
                right_child: Box::new(Node::Leaf { value: 4.0 }),
                missing_direction: MissingDirection::None,
            },
            weight: 1.0,
        };

        // Navigate: root -> left -> left -> left (should be leaf with value 1.0)
        match &tree.root {
            Node::Split { left_child, .. } => match &**left_child {
                Node::Split { left_child, .. } => match &**left_child {
                    Node::Split { left_child, .. } => match &**left_child {
                        Node::Leaf { value } => assert_eq!(*value, 1.0),
                        _ => panic!("Expected leaf at depth 3"),
                    },
                    _ => panic!("Expected split at depth 2"),
                },
                _ => panic!("Expected split at depth 1"),
            },
            _ => panic!("Expected split at root"),
        }
    }

    #[test]
    fn test_forest_with_weighted_trees() {
        // Test a forest with multiple trees of varying weights
        let forest = Forest {
            trees: vec![
                Tree {
                    root: Node::Leaf { value: 1.0 },
                    weight: 0.5,
                },
                Tree {
                    root: Node::Leaf { value: 2.0 },
                    weight: 1.0,
                },
                Tree {
                    root: Node::Leaf { value: 3.0 },
                    weight: 1.5,
                },
            ],
            base_score: 0.1,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };

        assert_eq!(forest.trees.len(), 3);
        assert_eq!(forest.trees[0].weight, 0.5);
        assert_eq!(forest.trees[1].weight, 1.0);
        assert_eq!(forest.trees[2].weight, 1.5);

        // Verify all trees are present
        for tree in &forest.trees {
            match tree.root {
                Node::Leaf { value } => {
                    assert!((1.0..=3.0).contains(&value));
                }
                _ => panic!("Expected leaf nodes"),
            }
        }
    }

    #[test]
    fn test_categorical_branch_direction_semantics() {
        // CRITICAL: Test that documents the categorical branch semantics
        // Value IN bitset should go RIGHT, value NOT in bitset should go LEFT
        // This is implemented in the backend as: left_cond = !bitset_contains(...)

        let tree = Tree {
            root: Node::Split {
                feature_idx: 0,
                split: SplitKind::Categorical {
                    bitoff: 0,
                    nbits: 4,
                    data: vec![0x05], // Binary: 00000101, categories 0 and 2 are IN the bitset
                },
                left_child: Box::new(Node::Leaf { value: 100.0 }), // NOT in bitset
                right_child: Box::new(Node::Leaf { value: 200.0 }), // IN bitset
                missing_direction: MissingDirection::Right,
            },
            weight: 1.0,
        };

        match tree.root {
            Node::Split {
                split,
                left_child,
                right_child,
                ..
            } => {
                match split {
                    SplitKind::Categorical { data, .. } => {
                        // Verify bitset encoding: bits 0 and 2 set
                        assert_eq!(data[0], 0x05);
                    }
                    _ => panic!("Expected categorical split"),
                }

                // Left child for values NOT in bitset (1, 3, etc.)
                match *left_child {
                    Node::Leaf { value } => assert_eq!(value, 100.0),
                    _ => panic!("Expected leaf"),
                }

                // Right child for values IN bitset (0, 2)
                match *right_child {
                    Node::Leaf { value } => assert_eq!(value, 200.0),
                    _ => panic!("Expected leaf"),
                }
            }
            _ => panic!("Expected split node"),
        }
    }
}
