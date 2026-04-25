use crate::ir::{
    AggregationKind, Forest, MissingDirection, Node, Operator, PostTransform, SplitKind, Tree,
};
use crate::onnx::ModelProto;
use anyhow::{anyhow, Context, Result};
use prost::Message;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct OnnxFrontend;

impl OnnxFrontend {
    pub fn new() -> Self {
        Self
    }
}

impl super::Frontend for OnnxFrontend {
    fn parse(&self, path: &Path) -> Result<Forest> {
        // 1. Read file
        let mut file =
            File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        // 2. Decode protobuf
        let model =
            ModelProto::decode(&*buffer).context("Failed to decode ONNX protobuf")?;
        let graph = model
            .graph
            .ok_or_else(|| anyhow!("ONNX Model has no graph"))?;

        // 3. Find TreeEnsemble operator
        let node = graph
            .node
            .iter()
            .find(|n| {
                matches!(
                    n.op_type.as_deref(),
                    Some("TreeEnsembleRegressor") | Some("TreeEnsembleClassifier")
                )
            })
            .ok_or_else(|| anyhow!("No TreeEnsemble operator found in ONNX graph"))?;

        let op_type = node.op_type.as_deref().unwrap_or("Unknown");
        println!("   > Found Operator: {}", op_type);

        // 4. Extract attributes
        let get_attr =
            |name: &str| node.attribute.iter().find(|a| a.name.as_deref() == Some(name));

        let empty_ints: Vec<i64> = vec![];
        let empty_floats: Vec<f32> = vec![];

        // --- Node structure (parallel arrays, one entry per node across all trees) ---
        let tree_ids = get_attr("nodes_treeids")
            .map(|a| &a.ints)
            .ok_or_else(|| anyhow!("Missing nodes_treeids"))?;
        let node_ids = get_attr("nodes_nodeids")
            .map(|a| &a.ints)
            .ok_or_else(|| anyhow!("Missing nodes_nodeids"))?;
        let feature_ids = get_attr("nodes_featureids")
            .map(|a| &a.ints)
            .ok_or_else(|| anyhow!("Missing nodes_featureids"))?;
        let values = get_attr("nodes_values")
            .map(|a| &a.floats)
            .ok_or_else(|| anyhow!("Missing nodes_values"))?;
        let modes = get_attr("nodes_modes")
            .map(|a| &a.strings)
            .ok_or_else(|| anyhow!("Missing nodes_modes"))?;
        let true_node_ids = get_attr("nodes_truenodeids")
            .map(|a| &a.ints)
            .ok_or_else(|| anyhow!("Missing nodes_truenodeids"))?;
        let false_node_ids = get_attr("nodes_falsenodeids")
            .map(|a| &a.ints)
            .ok_or_else(|| anyhow!("Missing nodes_falsenodeids"))?;

        // Bug 1 fix: per-node missing-value routing.
        // 1 = NaN follows the true_child (left in our IR), 0 = NaN follows false_child (right).
        let missing_tracks_true = get_attr("nodes_missing_value_tracks_true")
            .map(|a| &a.ints)
            .unwrap_or(&empty_ints);

        // --- Leaf values (regressor vs classifier) ---
        let target_tree_ids =
            get_attr("target_treeids").map(|a| &a.ints).unwrap_or(&empty_ints);
        let target_node_ids =
            get_attr("target_nodeids").map(|a| &a.ints).unwrap_or(&empty_ints);
        let target_weights =
            get_attr("target_weights").map(|a| &a.floats).unwrap_or(&empty_floats);

        let class_tree_ids =
            get_attr("class_treeids").map(|a| &a.ints).unwrap_or(&empty_ints);
        let class_node_ids =
            get_attr("class_nodeids").map(|a| &a.ints).unwrap_or(&empty_ints);
        let class_weights =
            get_attr("class_weights").map(|a| &a.floats).unwrap_or(&empty_floats);
        // Bug 2 fix: use class_ids to identify which weight belongs to which class.
        let class_ids = get_attr("class_ids")
            .map(|a| &a.ints)
            .unwrap_or(&empty_ints);

        // 5. Group nodes by tree, populating splits
        let mut trees_map: HashMap<i64, TreeBuilder> = HashMap::new();

        for i in 0..tree_ids.len() {
            let tid = tree_ids[i];
            let builder = trees_map.entry(tid).or_default();

            let mode_str =
                std::str::from_utf8(&modes[i]).unwrap_or("BRANCH_LEQ").to_string();

            // Bug 1 fix: read per-node flag; default false when attribute is absent.
            let tracks_true = missing_tracks_true.get(i).copied().unwrap_or(0) != 0;

            builder.splits.insert(
                node_ids[i],
                SplitInfo {
                    feature_idx: feature_ids[i] as usize,
                    threshold: values[i],
                    mode: mode_str,
                    true_child: true_node_ids[i],
                    false_child: false_node_ids[i],
                    missing_tracks_true: tracks_true,
                },
            );
        }

        // 6. Populate leaf values — regressor path
        for i in 0..target_tree_ids.len() {
            let builder = trees_map.entry(target_tree_ids[i]).or_default();
            builder
                .leaves
                .insert(target_node_ids[i], target_weights[i]);
        }

        // 7. Populate leaf values — classifier path (Bug 2 fix)
        //
        // class_ids, class_tree_ids, class_node_ids, class_weights are parallel arrays.
        // Each entry is one (tree, node, class) triple. We filter to the target class
        // instead of using fragile stride arithmetic.
        if !class_ids.is_empty() {
            let n_classes = class_ids.iter().max().map(|&m| (m + 1) as usize).unwrap_or(1);
            // Binary: take class 1 (positive). Single-output fallback: class 0.
            let target_class: i64 = if n_classes >= 2 { 1 } else { 0 };

            for i in 0..class_ids.len() {
                if class_ids[i] != target_class {
                    continue;
                }
                let tid = class_tree_ids[i];
                let nid = class_node_ids[i];
                let builder = trees_map.entry(tid).or_default();
                builder.leaves.insert(nid, class_weights[i]);
            }
        }

        // 8. Build recursive tree structures
        let mut sorted_tids: Vec<i64> = trees_map.keys().cloned().collect();
        sorted_tids.sort();

        let mut final_trees = Vec::new();
        for tid in &sorted_tids {
            let builder = &trees_map[tid];
            let root = build_recursive(0, builder)?;
            final_trees.push(Tree {
                root,
                weight: 1.0,
            });
        }

        // 9. Read ensemble-level attributes
        let aggregate_fn = get_attr("aggregate_function")
            .and_then(|a| a.s.as_ref())
            .and_then(|s| std::str::from_utf8(s).ok())
            .unwrap_or("SUM");

        let aggregation = match aggregate_fn {
            "AVERAGE" => AggregationKind::Average,
            _ => AggregationKind::Sum,
        };

        let onnx_post = get_attr("post_transform")
            .and_then(|a| a.s.as_ref())
            .and_then(|s| std::str::from_utf8(s).ok())
            .unwrap_or("NONE");

        let post_transform = match onnx_post {
            "LOGISTIC" => PostTransform::Logit,
            _ => PostTransform::Identity,
        };

        let base_score = get_attr("base_values")
            .and_then(|a| a.floats.first())
            .map(|&v| v as f64)
            .unwrap_or(0.0);

        println!(
            "   > {} trees | aggregation={:?} | post_transform={:?} | base_score={}",
            final_trees.len(),
            aggregation,
            post_transform,
            base_score
        );

        Ok(Forest {
            trees: final_trees,
            base_score,
            aggregation,
            post_transform,
        })
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[derive(Default)]
struct TreeBuilder {
    splits: HashMap<i64, SplitInfo>,
    leaves: HashMap<i64, f32>,
}

struct SplitInfo {
    feature_idx: usize,
    threshold: f32,
    mode: String,
    true_child: i64,
    false_child: i64,
    /// True when NaN should follow the true_child (= left in our IR).
    missing_tracks_true: bool,
}

/// Recursively build an `ir::Node` tree from the flat ONNX arrays.
fn build_recursive(node_id: i64, builder: &TreeBuilder) -> Result<Node> {
    // Leaf check first — leaves are terminal.
    if let Some(&val) = builder.leaves.get(&node_id) {
        return Ok(Node::Leaf { value: val });
    }

    let split = builder.splits.get(&node_id).ok_or_else(|| {
        anyhow!(
            "Node ID {} is neither a recorded split nor a leaf.",
            node_id
        )
    })?;

    // Defensive: mode == "LEAF" but no weight in leaves map.
    if split.mode == "LEAF" {
        return Ok(Node::Leaf { value: 0.0 });
    }

    // Recurse: true_child → left, false_child → right (our IR convention).
    let left = build_recursive(split.true_child, builder)?;
    let right = build_recursive(split.false_child, builder)?;

    let operator = match split.mode.as_str() {
        "BRANCH_LEQ" => Operator::LessOrEqual,
        "BRANCH_LT" => Operator::LessThan,
        "BRANCH_GT" => Operator::GreaterThan,
        "BRANCH_GEQ" => Operator::GreaterOrEqual,
        "BRANCH_EQ" => Operator::Equal,
        "BRANCH_NEQ" => Operator::NotEqual,
        _ => Operator::LessOrEqual, // safe default per ONNX spec
    };

    // Bug 1 fix: derive missing direction from per-node flag.
    let missing_direction = if split.missing_tracks_true {
        MissingDirection::Left // NaN → true_child → left
    } else {
        MissingDirection::Right // NaN → false_child → right
    };

    Ok(Node::Split {
        feature_idx: split.feature_idx,
        split: SplitKind::Numeric {
            threshold: split.threshold,
            operator,
        },
        left_child: Box::new(left),
        right_child: Box::new(right),
        missing_direction,
    })
}
