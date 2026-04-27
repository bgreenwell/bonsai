use crate::ir::{
    AggregationKind, Forest, MissingDirection, Node, Operator, PostTransform, SplitKind, Tree,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

pub struct LightgbmFrontend;

impl LightgbmFrontend {
    pub fn new() -> Self {
        Self
    }
}

impl super::Frontend for LightgbmFrontend {
    fn parse(&self, path: &Path) -> Result<Forest> {
        let content =
            fs::read_to_string(path).with_context(|| format!("Failed to read {:?}", path))?;
        let root: Value =
            serde_json::from_str(&content).context("Failed to parse LightGBM JSON")?;

        root.get("tree_info")
            .ok_or_else(|| anyhow!("Missing 'tree_info' key — not a LightGBM JSON model"))?;

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

        let post_transform = post_transform_for_objective(objective_name, num_tree_per_iteration);
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
            aggregation,
            post_transform,
        })
    }
}

// ---------------------------------------------------------------------------
// Objective mapping
// ---------------------------------------------------------------------------

fn post_transform_for_objective(name: &str, num_tree_per_iteration: usize) -> PostTransform {
    match name {
        "binary" | "cross_entropy" | "cross_entropy_lambda" | "binary_crossentropy" => {
            PostTransform::Logit
        }
        "multiclass" | "softmax" | "multiclassova" | "multiclass_ova" | "ovr" => {
            PostTransform::Softmax {
                n_classes: num_tree_per_iteration.max(2),
            }
        }
        _ => PostTransform::Identity,
    }
}

// ---------------------------------------------------------------------------
// Tree parsing
// ---------------------------------------------------------------------------

fn parse_tree(tv: &Value) -> Result<Tree> {
    let structure = tv
        .get("tree_structure")
        .ok_or_else(|| anyhow!("Missing 'tree_structure' in tree"))?;
    let root = parse_node(structure).context("tree_structure")?;
    // leaf values in LightGBM JSON are already scaled by shrinkage
    Ok(Tree { root, weight: 1.0 })
}

fn parse_node(node: &Value) -> Result<Node> {
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

    if decision_type == "==" {
        anyhow::bail!(
            "Categorical splits (decision_type '==') are not yet supported. \
             Feature index: {}",
            feature_idx
        );
    }

    // threshold may be a JSON string or number depending on LightGBM version
    let threshold: f32 = match &node["threshold"] {
        Value::Number(n) => {
            n.as_f64()
                .ok_or_else(|| anyhow!("threshold is not a valid float"))? as f32
        }
        Value::String(s) => s
            .parse::<f64>()
            .with_context(|| format!("Invalid threshold string: '{}'", s))? as f32,
        other => anyhow::bail!("Unexpected threshold type: {:?}", other),
    };

    let operator = match decision_type {
        "<=" => Operator::LessOrEqual,
        "<" => Operator::LessThan,
        ">" => Operator::GreaterThan,
        ">=" => Operator::GreaterOrEqual,
        other => anyhow::bail!("Unsupported decision_type: '{}'", other),
    };

    let default_left = node["default_left"].as_bool().unwrap_or(false);

    let left = parse_node(&node["left_child"]).context("left_child")?;
    let right = parse_node(&node["right_child"]).context("right_child")?;

    Ok(Node::Split {
        feature_idx,
        split: SplitKind::Numeric { threshold, operator },
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
    use crate::frontends::Frontend;
    use crate::ir::{MissingDirection, PostTransform};
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_json(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    fn parse(json: &str) -> Forest {
        let f = write_json(json);
        LightgbmFrontend::new().parse(f.path()).unwrap()
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

    // Threshold as JSON number (not string) — some LightGBM versions emit this
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
                split: SplitKind::Numeric { threshold, operator },
                left_child,
                right_child,
                ..
            } => (*feature_idx, *threshold, operator.clone(), left_child, right_child),
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
        assert_eq!(forest.post_transform, PostTransform::Softmax { n_classes: 3 });
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
        let f = write_json(r#"{"not_tree_info": []}"#);
        assert!(LightgbmFrontend::new().parse(f.path()).is_err());
    }
}
