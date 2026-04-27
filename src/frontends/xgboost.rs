use crate::ir::{
    AggregationKind, Forest, MissingDirection, Node, Operator, PostTransform, SplitKind, Tree,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::fs;
use std::path::Path;

pub struct XgboostFrontend;

impl XgboostFrontend {
    pub fn new() -> Self {
        Self
    }
}

impl super::Frontend for XgboostFrontend {
    fn parse(&self, path: &Path) -> Result<Forest> {
        let content =
            fs::read_to_string(path).with_context(|| format!("Failed to read {:?}", path))?;
        let root: Value = serde_json::from_str(&content).context("Failed to parse XGBoost JSON")?;

        let learner = root
            .get("learner")
            .ok_or_else(|| anyhow!("Missing 'learner' key — not an XGBoost JSON model"))?;

        // --- Model-level parameters ---
        let model_param = &learner["learner_model_param"];

        let base_score_str = model_param["base_score"].as_str().unwrap_or("0.5");
        // XGBoost 3.x wraps base_score in brackets. For binary/regression: "[5E-1]" → scalar.
        // For multiclass: "[v0,v1,v2]" → one value per class.
        let base_score_inner = base_score_str.trim_matches(|c| c == '[' || c == ']');

        let num_class: usize = model_param["num_class"]
            .as_str()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // --- Objective → post-transform ---
        let objective_name = learner["objective"]["name"]
            .as_str()
            .unwrap_or("reg:squarederror");
        println!("   > objective: {}", objective_name);

        let post_transform = post_transform_for_objective(objective_name, num_class);

        // Parse base_score: scalar for binary/regression, vector for multiclass.
        let (base_score, base_scores) = if base_score_inner.contains(',') {
            // Multiclass vector: "[v0,v1,v2]" → Vec<f64>
            let scores: Vec<f64> = base_score_inner
                .split(',')
                .map(|s| s.trim().parse::<f64>())
                .collect::<std::result::Result<_, _>>()
                .with_context(|| {
                    format!("Invalid multiclass base_score vector: '{}'", base_score_str)
                })?;
            (0.0f64, scores)
        } else {
            // Scalar: XGBoost may store in probability space for logistic objectives.
            let raw: f64 = base_score_inner
                .parse()
                .with_context(|| format!("Invalid base_score: '{}'", base_score_str))?;
            let converted = if matches!(post_transform, PostTransform::Logit)
                && raw > 0.0
                && raw < 1.0
            {
                (raw / (1.0 - raw)).ln()
            } else {
                raw
            };
            (converted, vec![])
        };

        // --- Parse trees ---
        let trees_val = &learner["gradient_booster"]["model"]["trees"];
        let trees_arr = trees_val
            .as_array()
            .ok_or_else(|| anyhow!("learner.gradient_booster.model.trees is not an array"))?;

        println!("   > {} trees", trees_arr.len());

        let trees: Vec<Tree> = trees_arr
            .iter()
            .enumerate()
            .map(|(i, tv)| parse_tree(tv).with_context(|| format!("tree {}", i)))
            .collect::<Result<_>>()?;

        Ok(Forest {
            trees,
            base_score,
            base_scores,
            aggregation: AggregationKind::Sum,
            post_transform,
        })
    }
}

// ---------------------------------------------------------------------------
// Objective mapping
// ---------------------------------------------------------------------------

fn post_transform_for_objective(name: &str, num_class: usize) -> PostTransform {
    match name {
        "binary:logistic" | "reg:logistic" => PostTransform::Logit,
        // binary:logitraw outputs raw margin (no sigmoid); sigmoid is the caller's responsibility
        "binary:logitraw" => PostTransform::Identity,
        "multi:softmax" | "multi:softprob" => PostTransform::Softmax {
            n_classes: num_class.max(2),
        },
        _ => PostTransform::Identity,
    }
}

// ---------------------------------------------------------------------------
// Tree parsing
// ---------------------------------------------------------------------------

const MAX_TREE_DEPTH: usize = 256;

fn parse_tree(tv: &Value) -> Result<Tree> {
    let left = int_array(&tv["left_children"]).context("left_children")?;
    let right = int_array(&tv["right_children"]).context("right_children")?;
    let feat = int_array(&tv["split_indices"]).context("split_indices")?;
    let cond = float_array(&tv["split_conditions"]).context("split_conditions")?;
    let def_left = int_array(&tv["default_left"]).context("default_left")?;

    anyhow::ensure!(
        left.len() == right.len()
            && left.len() == feat.len()
            && left.len() == cond.len()
            && left.len() == def_left.len(),
        "XGBoost tree arrays have mismatched lengths"
    );

    let root = build_node(0, &left, &right, &feat, &cond, &def_left, 0)?;
    Ok(Tree { root, weight: 1.0 })
}

fn build_node(
    id: i64,
    left: &[i64],
    right: &[i64],
    feat: &[i64],
    cond: &[f32],
    def_left: &[i64],
    depth: usize,
) -> Result<Node> {
    anyhow::ensure!(
        depth <= MAX_TREE_DEPTH,
        "tree depth exceeds maximum ({MAX_TREE_DEPTH}); possible malformed model"
    );
    let uid = id as usize;
    anyhow::ensure!(uid < left.len(), "node id {} out of bounds", id);

    if left[uid] == -1 {
        return Ok(Node::Leaf {
            value: cond[uid] as f64,
        });
    }

    anyhow::ensure!(
        left[uid] >= 0 && right[uid] >= 0,
        "node {} has invalid child id (left={}, right={})",
        id,
        left[uid],
        right[uid]
    );

    let l = build_node(left[uid], left, right, feat, cond, def_left, depth + 1)?;
    let r = build_node(right[uid], left, right, feat, cond, def_left, depth + 1)?;

    // XGBoost split: feature[i] < threshold → yes → left child
    Ok(Node::Split {
        feature_idx: feat[uid] as usize,
        split: SplitKind::Numeric {
            threshold: cond[uid],
            operator: Operator::LessThan,
        },
        left_child: Box::new(l),
        right_child: Box::new(r),
        missing_direction: if def_left[uid] != 0 {
            MissingDirection::Left
        } else {
            MissingDirection::Right
        },
    })
}

// ---------------------------------------------------------------------------
// JSON array helpers
// ---------------------------------------------------------------------------

fn int_array(val: &Value) -> Result<Vec<i64>> {
    val.as_array()
        .ok_or_else(|| anyhow!("expected array, got {:?}", val.as_str().unwrap_or("?")))?
        .iter()
        .map(|v| v.as_i64().ok_or_else(|| anyhow!("expected integer, got {:?}", v)))
        .collect()
}

fn float_array(val: &Value) -> Result<Vec<f32>> {
    val.as_array()
        .ok_or_else(|| anyhow!("expected array, got {:?}", val.as_str().unwrap_or("?")))?
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|x| x as f32)
                .ok_or_else(|| anyhow!("expected float, got {:?}", v))
        })
        .collect()
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
        XgboostFrontend::new().parse(f.path()).unwrap()
    }

    // Minimal 3-node tree: root splits feature[0] < 0.5; left leaf=1.0, right leaf=-1.0
    const REGRESSION_JSON: &str = r#"{
      "learner": {
        "learner_model_param": {"base_score":"0","num_class":"0","num_feature":"2"},
        "objective": {"name": "reg:squarederror"},
        "gradient_booster": {"model": {"trees": [{
          "left_children":  [1, -1, -1],
          "right_children": [2, -1, -1],
          "split_indices":  [0,  0,  0],
          "split_conditions": [0.5, 1.0, -1.0],
          "default_left":   [0,  0,  0]
        }]}}
      }
    }"#;

    // Same tree with binary:logistic objective and feature[1] as root, NaN→left
    const LOGISTIC_JSON: &str = r#"{
      "learner": {
        "learner_model_param": {"base_score":"0","num_class":"0","num_feature":"2"},
        "objective": {"name": "binary:logistic"},
        "gradient_booster": {"model": {"trees": [{
          "left_children":  [1, -1, -1],
          "right_children": [2, -1, -1],
          "split_indices":  [1,  0,  0],
          "split_conditions": [0.3, 0.5, -0.5],
          "default_left":   [1,  0,  0]
        }]}}
      }
    }"#;

    const MULTICLASS_JSON: &str = r#"{
      "learner": {
        "learner_model_param": {"base_score":"0","num_class":"3","num_feature":"4"},
        "objective": {"name": "multi:softprob"},
        "gradient_booster": {"model": {"trees": [
          {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.1],"default_left":[0]},
          {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.2],"default_left":[0]},
          {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.3],"default_left":[0]},
          {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.4],"default_left":[0]},
          {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.5],"default_left":[0]},
          {"left_children":[-1],"right_children":[-1],"split_indices":[0],"split_conditions":[0.6],"default_left":[0]}
        ]}}
      }
    }"#;

    #[test]
    fn test_regression_tree_structure() {
        let forest = parse(REGRESSION_JSON);

        assert_eq!(forest.trees.len(), 1);
        assert_eq!(forest.post_transform, PostTransform::Identity);
        assert_eq!(forest.base_score, 0.0);

        // Root: split on feature 0, threshold 0.5, operator LessThan
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
        assert_eq!(op, Operator::LessThan);

        // Left leaf = 1.0, right leaf = -1.0
        assert!(matches!(**left,  Node::Leaf { value } if (value - 1.0).abs() < 1e-6));
        assert!(matches!(**right, Node::Leaf { value } if (value + 1.0).abs() < 1e-6));
    }

    #[test]
    fn test_logistic_post_transform_and_nan_routing() {
        let forest = parse(LOGISTIC_JSON);

        assert_eq!(forest.post_transform, PostTransform::Logit);
        assert_eq!(forest.base_score, 0.0);

        // Root split: feature 1, NaN → left (default_left = 1)
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
    fn test_base_score_logit_conversion() {
        // "5e-01" = 0.5 → logit(0.5) = 0.0
        let json = LOGISTIC_JSON.replace(r#""base_score":"0""#, r#""base_score":"5e-01""#);
        let forest = parse(&json);
        assert!(
            forest.base_score.abs() < 1e-9,
            "logit(0.5) should be ~0, got {}",
            forest.base_score
        );
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
    fn test_missing_learner_key_errors() {
        let f = write_json(r#"{"not_learner": {}}"#);
        assert!(XgboostFrontend::new().parse(f.path()).is_err());
    }
}
