use crate::ir::{
    AggregationKind, Forest, MissingDirection, Node, Operator, PostTransform, SplitKind, Tree,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
pub(crate) fn parse_json(root: &Value) -> Result<Forest> {
    root.get("tree_info")
        .ok_or_else(|| anyhow!("Missing 'tree_info' key - not a LightGBM JSON model"))?;

    // --- Model-level parameters ---
    let num_tree_per_iteration: usize =
        root["num_tree_per_iteration"].as_u64().unwrap_or(1) as usize;

    // objective may be a JSON array ["binary", "crossentropy"] or a
    // space-separated string "binary sigmoid:1" depending on LightGBM version
    let objective_name_owned: String;
    let objective_name = match &root["objective"] {
        serde_json::Value::Array(arr) => {
            objective_name_owned = arr
                .first()
                .and_then(|v| v.as_str())
                .unwrap_or("regression")
                .to_string();
            objective_name_owned.as_str()
        }
        serde_json::Value::String(s) => {
            objective_name_owned = s
                .split_whitespace()
                .next()
                .unwrap_or("regression")
                .to_string();
            objective_name_owned.as_str()
        }
        _ => "regression",
    };
    println!("   > objective: {}", objective_name);

    let average_output = root["average_output"].as_bool().unwrap_or(false);

    let post_transform = post_transform_for_objective(objective_name, num_tree_per_iteration)?;
    let aggregation = if average_output {
        AggregationKind::Average
    } else {
        AggregationKind::Sum
    };

    // --- Parse trees ---
    let trees_arr = root["tree_info"]
        .as_array()
        .ok_or_else(|| anyhow!("'tree_info' is not an array"))?;

    println!("   > {} trees", trees_arr.len());

    let trees: Vec<Tree> = trees_arr
        .iter()
        .enumerate()
        .map(|(i, tv)| parse_tree(tv).with_context(|| format!("tree {}", i)))
        .collect::<Result<_>>()?;

    Ok(Forest {
        trees,
        base_score: 0.0,
        base_scores: vec![],
        aggregation,
        post_transform,
        catboost_metadata: None,
    })
}

// ---------------------------------------------------------------------------
// Objective mapping
// ---------------------------------------------------------------------------

// Mappings verified against LightGBM's own predict() vs raw_score output.
fn post_transform_for_objective(
    name: &str,
    num_tree_per_iteration: usize,
) -> Result<PostTransform> {
    Ok(match name {
        "binary" | "cross_entropy" | "binary_crossentropy" | "xentropy" => PostTransform::Logit,
        "multiclass" | "softmax" => PostTransform::Softmax {
            n_classes: num_tree_per_iteration.max(2),
        },
        // One-vs-all applies a per-class sigmoid, not softmax, which the IR
        // cannot express yet.
        "multiclassova" | "multiclass_ova" | "ova" | "ovr" => anyhow::bail!(
            "multiclassova applies a per-class sigmoid, which bonsai does not \
             implement; retrain with objective=multiclass"
        ),
        // cross_entropy_lambda's output transform is neither sigmoid nor a
        // log link; reject rather than mis-predict.
        "cross_entropy_lambda" | "xentlambda" => anyhow::bail!(
            "cross_entropy_lambda uses an output transform bonsai does not \
             implement; retrain with objective=cross_entropy"
        ),
        // Log-link objectives: LightGBM applies exp() at prediction time.
        "poisson" | "gamma" | "tweedie" => PostTransform::Log,
        // Raw-score objectives, including ranking.
        "regression"
        | "regression_l2"
        | "l2"
        | "mean_squared_error"
        | "mse"
        | "l2_root"
        | "root_mean_squared_error"
        | "rmse"
        | "regression_l1"
        | "l1"
        | "mean_absolute_error"
        | "mae"
        | "huber"
        | "fair"
        | "quantile"
        | "mape"
        | "mean_absolute_percentage_error"
        | "lambdarank"
        | "rank_xendcg"
        | "xendcg"
        | "xe_ndcg"
        | "xe_ndcg_mart"
        | "xendcg_mart" => PostTransform::Identity,
        other => {
            println!(
                "   ! Unknown objective '{}', emitting raw scores (no post-transform)",
                other
            );
            PostTransform::Identity
        }
    })
}

// ---------------------------------------------------------------------------
// Tree parsing
// ---------------------------------------------------------------------------

fn parse_tree(tv: &Value) -> Result<Tree> {
    let structure = tv
        .get("tree_structure")
        .ok_or_else(|| anyhow!("Missing 'tree_structure' in tree"))?;
    let root = parse_node(structure, 0).context("tree_structure")?;
    // leaf values in LightGBM JSON are already scaled by shrinkage
    Ok(Tree { root, weight: 1.0 })
}

const MAX_TREE_DEPTH: usize = 256;

fn parse_node(node: &Value, depth: usize) -> Result<Node> {
    anyhow::ensure!(
        depth <= MAX_TREE_DEPTH,
        "tree depth exceeds maximum ({MAX_TREE_DEPTH}); possible malformed model"
    );
    if let Some(v) = node.get("leaf_value") {
        let value = v
            .as_f64()
            .ok_or_else(|| anyhow!("leaf_value is not a number: {:?}", v))?;
        return Ok(Node::Leaf { value });
    }

    let feature_idx = node["split_feature"]
        .as_u64()
        .ok_or_else(|| anyhow!("Missing or invalid split_feature"))? as usize;

    let decision_type = node["decision_type"].as_str().unwrap_or("<=");

    let default_left = node["default_left"].as_bool().unwrap_or(false);
    let left = parse_node(&node["left_child"], depth + 1).context("left_child")?;
    let right = parse_node(&node["right_child"], depth + 1).context("right_child")?;

    if decision_type == "==" {
        // LightGBM categorical split: threshold is a "||"-separated list of category indices
        // (e.g. "0||2||5") that go to the LEFT child.
        //
        // Bonsai IR convention: value IN bitset → RIGHT child.
        // To reconcile: store the threshold set in the bitset, then swap left/right children
        // so that IN-set rows go RIGHT (the swapped original left).
        // Missing direction also flips because the children are swapped.
        let threshold_str = match &node["threshold"] {
            Value::String(s) => s.clone(),
            Value::Number(n) => n.to_string(),
            other => anyhow::bail!("Unexpected categorical threshold type: {:?}", other),
        };

        let categories: Vec<u32> = threshold_str
            .split("||")
            .map(|s| s.trim().parse::<u32>())
            .collect::<std::result::Result<_, _>>()
            .with_context(|| {
                format!(
                    "Invalid categorical threshold '{}': expected pipe-separated integers",
                    threshold_str
                )
            })?;

        let max_cat = categories.iter().max().copied().unwrap_or(0);
        let nbits = max_cat + 1;
        let nbytes = ((nbits.saturating_sub(1)) / 8 + 1) as usize;
        let mut data = vec![0u8; nbytes];
        for &cat in &categories {
            data[(cat / 8) as usize] |= 1u8 << (cat % 8);
        }

        return Ok(Node::Split {
            feature_idx,
            split: SplitKind::Categorical {
                bitoff: 0,
                nbits,
                data,
            },
            // Swap: original left (IN set) becomes right; original right becomes left.
            left_child: Box::new(right),
            right_child: Box::new(left),
            // Missing direction also flips because children are swapped.
            missing_direction: if default_left {
                MissingDirection::Right
            } else {
                MissingDirection::Left
            },
        });
    }

    // threshold may be a JSON string or number depending on LightGBM version
    let threshold: f32 = match &node["threshold"] {
        Value::Number(n) => {
            n.as_f64()
                .ok_or_else(|| anyhow!("threshold is not a valid float"))? as f32
        }
        Value::String(s) => {
            s.parse::<f64>()
                .with_context(|| format!("Invalid threshold string: '{}'", s))? as f32
        }
        other => anyhow::bail!("Unexpected threshold type: {:?}", other),
    };

    let operator = match decision_type {
        "<=" => Operator::LessOrEqual,
        "<" => Operator::LessThan,
        ">" => Operator::GreaterThan,
        ">=" => Operator::GreaterOrEqual,
        other => anyhow::bail!("Unsupported decision_type: '{}'", other),
    };

    Ok(Node::Split {
        feature_idx,
        split: SplitKind::Numeric {
            threshold,
            operator,
        },
        left_child: Box::new(left),
        right_child: Box::new(right),
        missing_direction: if default_left {
            MissingDirection::Left
        } else {
            MissingDirection::Right
        },
    })
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{MissingDirection, PostTransform};

    fn parse(json: &str) -> Forest {
        let root: serde_json::Value = serde_json::from_str(json).unwrap();
        parse_json(&root).unwrap()
    }

    // Minimal regression tree: root splits feature[0] <= 0.5; left leaf=1.0, right leaf=-1.0
    const REGRESSION_JSON: &str = r#"{
      "name": "tree",
      "num_class": 1,
      "num_tree_per_iteration": 1,
      "objective": ["regression", "mean_squared_error"],
      "average_output": false,
      "tree_info": [{
        "num_leaves": 2,
        "shrinkage": 1.0,
        "tree_structure": {
          "split_index": 0,
          "split_feature": 0,
          "split_gain": 100.0,
          "threshold": "0.5",
          "decision_type": "<=",
          "default_left": false,
          "left_child":  {"leaf_index": 0, "leaf_value":  1.0},
          "right_child": {"leaf_index": 1, "leaf_value": -1.0}
        }
      }]
    }"#;

    // Binary classification with NaN routing to left on feature[1]
    const LOGISTIC_JSON: &str = r#"{
      "name": "tree",
      "num_class": 1,
      "num_tree_per_iteration": 1,
      "objective": ["binary", "crossentropy"],
      "average_output": false,
      "tree_info": [{
        "num_leaves": 2,
        "shrinkage": 1.0,
        "tree_structure": {
          "split_index": 0,
          "split_feature": 1,
          "split_gain": 50.0,
          "threshold": "0.3",
          "decision_type": "<=",
          "default_left": true,
          "left_child":  {"leaf_index": 0, "leaf_value":  0.5},
          "right_child": {"leaf_index": 1, "leaf_value": -0.5}
        }
      }]
    }"#;

    // Multiclass (3 classes, 6 trees)
    const MULTICLASS_JSON: &str = r#"{
      "name": "tree",
      "num_class": 3,
      "num_tree_per_iteration": 3,
      "objective": ["multiclass", "softmax"],
      "average_output": false,
      "tree_info": [
        {"num_leaves":1,"shrinkage":1.0,"tree_structure":{"leaf_index":0,"leaf_value":0.1}},
        {"num_leaves":1,"shrinkage":1.0,"tree_structure":{"leaf_index":0,"leaf_value":0.2}},
        {"num_leaves":1,"shrinkage":1.0,"tree_structure":{"leaf_index":0,"leaf_value":0.3}},
        {"num_leaves":1,"shrinkage":1.0,"tree_structure":{"leaf_index":0,"leaf_value":0.4}},
        {"num_leaves":1,"shrinkage":1.0,"tree_structure":{"leaf_index":0,"leaf_value":0.5}},
        {"num_leaves":1,"shrinkage":1.0,"tree_structure":{"leaf_index":0,"leaf_value":0.6}}
      ]
    }"#;

    // Threshold as JSON number (not string) - some LightGBM versions emit this
    const NUMERIC_THRESHOLD_JSON: &str = r#"{
      "name": "tree",
      "num_class": 1,
      "num_tree_per_iteration": 1,
      "objective": ["regression", "mse"],
      "average_output": false,
      "tree_info": [{
        "num_leaves": 2,
        "shrinkage": 1.0,
        "tree_structure": {
          "split_index": 0,
          "split_feature": 2,
          "split_gain": 10.0,
          "threshold": 0.75,
          "decision_type": "<=",
          "default_left": false,
          "left_child":  {"leaf_index": 0, "leaf_value": 2.0},
          "right_child": {"leaf_index": 1, "leaf_value": 3.0}
        }
      }]
    }"#;

    #[test]
    fn test_regression_tree_structure() {
        let forest = parse(REGRESSION_JSON);

        assert_eq!(forest.trees.len(), 1);
        assert_eq!(forest.post_transform, PostTransform::Identity);
        assert_eq!(forest.base_score, 0.0);

        let root = &forest.trees[0].root;
        let (fi, th, op, left, right) = match root {
            Node::Split {
                feature_idx,
                split:
                    SplitKind::Numeric {
                        threshold,
                        operator,
                    },
                left_child,
                right_child,
                ..
            } => (
                *feature_idx,
                *threshold,
                operator.clone(),
                left_child,
                right_child,
            ),
            _ => panic!("expected numeric split at root"),
        };

        assert_eq!(fi, 0);
        assert!((th - 0.5).abs() < 1e-6, "threshold {}", th);
        assert_eq!(op, Operator::LessOrEqual);

        assert!(matches!(**left,  Node::Leaf { value } if (value - 1.0).abs() < 1e-6));
        assert!(matches!(**right, Node::Leaf { value } if (value + 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_logistic_post_transform_and_nan_routing() {
        let forest = parse(LOGISTIC_JSON);

        assert_eq!(forest.post_transform, PostTransform::Logit);
        assert_eq!(forest.base_score, 0.0);

        match &forest.trees[0].root {
            Node::Split {
                feature_idx,
                missing_direction,
                ..
            } => {
                assert_eq!(*feature_idx, 1);
                assert_eq!(*missing_direction, MissingDirection::Left);
            }
            _ => panic!("expected split"),
        }
    }

    #[test]
    fn test_multiclass_softmax() {
        let forest = parse(MULTICLASS_JSON);
        assert_eq!(
            forest.post_transform,
            PostTransform::Softmax { n_classes: 3 }
        );
        assert_eq!(forest.trees.len(), 6);
    }

    #[test]
    fn test_numeric_threshold_as_json_number() {
        let forest = parse(NUMERIC_THRESHOLD_JSON);
        assert_eq!(forest.trees.len(), 1);
        match &forest.trees[0].root {
            Node::Split {
                feature_idx,
                split: SplitKind::Numeric { threshold, .. },
                ..
            } => {
                assert_eq!(*feature_idx, 2);
                assert!((threshold - 0.75).abs() < 1e-6, "threshold {}", threshold);
            }
            _ => panic!("expected numeric split"),
        }
    }

    #[test]
    fn test_missing_tree_info_key_errors() {
        let root: serde_json::Value = serde_json::from_str(r#"{"not_tree_info": []}"#).unwrap();
        assert!(parse_json(&root).is_err());
    }

    // Categorical split: decision_type "==" with threshold "0||2" (categories 0 and 2 go left)
    const CATEGORICAL_JSON: &str = r#"{
      "name": "tree",
      "num_class": 1,
      "num_tree_per_iteration": 1,
      "objective": ["regression", "mean_squared_error"],
      "average_output": false,
      "tree_info": [{
        "num_leaves": 2,
        "shrinkage": 1.0,
        "tree_structure": {
          "split_index": 0,
          "split_feature": 3,
          "split_gain": 80.0,
          "threshold": "0||2",
          "decision_type": "==",
          "default_left": false,
          "left_child":  {"leaf_index": 0, "leaf_value": 5.0},
          "right_child": {"leaf_index": 1, "leaf_value": -5.0}
        }
      }]
    }"#;

    #[test]
    fn test_categorical_split_parses_bitset() {
        let forest = parse(CATEGORICAL_JSON);
        assert_eq!(forest.trees.len(), 1);

        match &forest.trees[0].root {
            Node::Split {
                feature_idx,
                split:
                    SplitKind::Categorical {
                        bitoff,
                        nbits,
                        data,
                    },
                left_child,
                right_child,
                missing_direction,
            } => {
                assert_eq!(*feature_idx, 3);
                assert_eq!(*bitoff, 0);
                assert_eq!(*nbits, 3); // max category is 2, so nbits = 3
                                       // categories 0 and 2 are set: bits 0 and 2 → 0b00000101 = 0x05
                assert_eq!(data[0], 0x05, "bitset byte should have bits 0 and 2 set");

                // After child-swap: bonsai IN-bitset → right (original left = 5.0)
                //                   NOT in bitset   → left  (original right = -5.0)
                match left_child.as_ref() {
                    Node::Leaf { value } => assert!(
                        (value + 5.0).abs() < 1e-9,
                        "left should be original right child (-5.0), got {}",
                        value
                    ),
                    _ => panic!("expected leaf"),
                }
                match right_child.as_ref() {
                    Node::Leaf { value } => assert!(
                        (value - 5.0).abs() < 1e-9,
                        "right should be original left child (5.0), got {}",
                        value
                    ),
                    _ => panic!("expected leaf"),
                }

                // default_left=false → original missing goes right (original right = -5.0)
                // After swap: missing goes LEFT (the swapped right = original left = -5.0)
                assert_eq!(*missing_direction, MissingDirection::Left);
            }
            _ => panic!("expected categorical split"),
        }
    }

    #[test]
    fn test_categorical_split_single_category() {
        let json = r#"{
          "name": "tree",
          "num_class": 1,
          "num_tree_per_iteration": 1,
          "objective": ["regression", "mse"],
          "average_output": false,
          "tree_info": [{
            "num_leaves": 2,
            "shrinkage": 1.0,
            "tree_structure": {
              "split_index": 0, "split_feature": 1, "split_gain": 10.0,
              "threshold": "7",
              "decision_type": "==",
              "default_left": true,
              "left_child":  {"leaf_index": 0, "leaf_value": 1.0},
              "right_child": {"leaf_index": 1, "leaf_value": 2.0}
            }
          }]
        }"#;
        let forest = parse(json);
        match &forest.trees[0].root {
            Node::Split {
                split: SplitKind::Categorical { nbits, data, .. },
                missing_direction,
                ..
            } => {
                assert_eq!(*nbits, 8); // category 7 → bit 7 of first byte
                assert_eq!(data[0], 0x80, "bit 7 should be set: 0x80");
                // default_left=true → after swap → MissingDirection::Right
                assert_eq!(*missing_direction, MissingDirection::Right);
            }
            _ => panic!("expected categorical split"),
        }
    }

    #[test]
    fn test_categorical_split_invalid_threshold_errors() {
        let json = r#"{
          "name": "tree",
          "num_class": 1,
          "num_tree_per_iteration": 1,
          "objective": ["regression", "mse"],
          "average_output": false,
          "tree_info": [{
            "num_leaves": 2,
            "shrinkage": 1.0,
            "tree_structure": {
              "split_index": 0, "split_feature": 0, "split_gain": 10.0,
              "threshold": "not_a_number",
              "decision_type": "==",
              "default_left": false,
              "left_child":  {"leaf_index": 0, "leaf_value": 1.0},
              "right_child": {"leaf_index": 1, "leaf_value": 2.0}
            }
          }]
        }"#;
        let root: serde_json::Value = serde_json::from_str(json).unwrap();
        assert!(
            parse_json(&root).is_err(),
            "should fail on non-integer categorical threshold"
        );
    }

    #[test]
    fn test_log_link_and_ranking_objectives() {
        let poisson = parse(
            &REGRESSION_JSON.replace(r#"["regression", "mean_squared_error"]"#, r#"["poisson"]"#),
        );
        assert_eq!(poisson.post_transform, PostTransform::Log);

        let rank = parse(&REGRESSION_JSON.replace(
            r#"["regression", "mean_squared_error"]"#,
            r#"["lambdarank"]"#,
        ));
        assert_eq!(rank.post_transform, PostTransform::Identity);
    }

    #[test]
    fn test_unsupported_transforms_rejected() {
        for obj in ["multiclassova", "cross_entropy_lambda"] {
            let json = REGRESSION_JSON.replace(
                r#"["regression", "mean_squared_error"]"#,
                &format!(r#"["{}"]"#, obj),
            );
            let root: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert!(
                parse_json(&root).is_err(),
                "objective '{}' should be rejected",
                obj
            );
        }
    }
}
