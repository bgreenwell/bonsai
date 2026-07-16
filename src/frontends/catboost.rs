use crate::ir::{
    AggregationKind, CatboostMetadata, CtrElement, CtrInfo, CtrValueTable, Forest,
    MissingDirection, Node, Operator, PostTransform, SplitKind, Tree,
};
use anyhow::{anyhow, Context, Result};
use serde_json::Value;
use std::collections::HashMap;
pub(crate) fn parse_json(root: &Value) -> Result<Forest> {
    let model_info = &root["model_info"];

    let mut classes_count = model_info["params"]["data_processing_options"]["classes_count"]
        .as_u64()
        .or_else(|| model_info["params"]["loss_function"]["params"]["classes_count"].as_u64())
        .unwrap_or(1) as usize;

    if classes_count == 0 {
        classes_count = 1;
    }

    // Loss function location varies by CatBoost version: older exports use
    // model_info.loss_function, newer ones model_info.params.loss_function.
    let loss_function = model_info["loss_function"]["type"]
        .as_str()
        .or_else(|| model_info["loss_function"].as_str())
        .or_else(|| model_info["params"]["loss_function"]["type"].as_str())
        .or_else(|| model_info["params"]["loss_function"].as_str())
        .unwrap_or("RMSE");

    println!("   > loss function: {}", loss_function);
    if classes_count > 1 {
        println!("   > classes: {}", classes_count);
    }

    let post_transform = match loss_function {
        "Logloss" | "CrossEntropy" => PostTransform::Logit,
        "MultiClass" | "MultiClassOneVsAll" => PostTransform::Softmax {
            n_classes: classes_count,
        },
        "RMSE" | "MAE" | "Quantile" | "Poisson" => PostTransform::Identity,
        _ => {
            println!(
                "   ! Unknown loss function '{}', defaulting to Identity",
                loss_function
            );
            PostTransform::Identity
        }
    };

    // --- CTR metadata (only populated for models with categorical features) ---
    let mut ctrs = Vec::new();
    if let Some(ctrs_arr) = root["features_info"]["ctrs"].as_array() {
        for c_val in ctrs_arr {
            let mut elements = Vec::new();
            if let Some(elem_arr) = c_val["elements"].as_array() {
                for e_val in elem_arr {
                    elements.push(CtrElement {
                        cat_feature_index: e_val["cat_feature_index"].as_u64().unwrap_or(0)
                            as usize,
                        combination_element: e_val["combination_element"]
                            .as_str()
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }
            ctrs.push(CtrInfo {
                elements,
                identifier: c_val["identifier"].as_str().unwrap_or("").to_string(),
                prior_numerator: c_val["prior_numerator"].as_f64().unwrap_or(0.0),
                // CatBoost's JSON format spells this key with the historic typo.
                prior_denominator: c_val["prior_denomerator"].as_f64().unwrap_or(1.0),
                scale: c_val["scale"].as_f64().unwrap_or(1.0),
                shift: c_val["shift"].as_f64().unwrap_or(0.0),
            });
        }
    }

    let mut ctr_data = HashMap::new();
    if let Some(cd_map) = root["ctr_data"].as_object() {
        for (id, val) in cd_map {
            let stride = val["hash_stride"].as_u64().unwrap_or(1) as usize;
            let mut hash_map: Vec<u64> = val["hash_map"]
                .as_array()
                .unwrap_or(&vec![])
                .iter()
                .map(|v| {
                    if let Some(s) = v.as_str() {
                        s.parse::<u64>().unwrap_or(0)
                    } else {
                        v.as_u64().unwrap_or(0)
                    }
                })
                .collect();

            // CatBoost JSON map might not be sorted; we need it sorted for binary search.
            let mut chunks: Vec<Vec<u64>> =
                hash_map.chunks_exact(stride).map(|c| c.to_vec()).collect();
            chunks.sort_by_key(|c| c[0]);
            hash_map = chunks.into_iter().flatten().collect();

            ctr_data.insert(
                id.clone(),
                CtrValueTable {
                    hash_map,
                    hash_stride: stride,
                    counter_denominator: val["counter_denominator"].as_i64().unwrap_or(0),
                },
            );
        }
    }

    // Only attach CatBoost metadata when the model actually uses categorical features.
    let catboost_metadata = if ctrs.is_empty() {
        None
    } else {
        Some(CatboostMetadata { ctrs, ctr_data })
    };

    // Parse oblivious trees
    let trees_val = root
        .get("oblivious_trees")
        .ok_or_else(|| anyhow!("Missing 'oblivious_trees' key — not a CatBoost JSON model"))?;

    let trees_arr = trees_val
        .as_array()
        .ok_or_else(|| anyhow!("'oblivious_trees' is not an array"))?;

    println!("   > {} trees in JSON", trees_arr.len());

    let mut all_trees = Vec::new();
    for (i, tv) in trees_arr.iter().enumerate() {
        let multi_trees =
            parse_oblivious_tree(tv, classes_count).with_context(|| format!("tree {}", i))?;
        all_trees.extend(multi_trees);
    }

    println!("   > {} IR trees", all_trees.len());

    let (scale, biases) = if let Some(sb_arr) = root["scale_and_bias"].as_array() {
        let scale = sb_arr[0].as_f64().unwrap_or(1.0);
        let biases: Vec<f64> = sb_arr[1]
            .as_array()
            .unwrap_or(&vec![])
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0))
            .collect();
        (scale, biases)
    } else {
        (1.0, vec![0.0])
    };

    if scale != 1.0 {
        println!("   > Applying scale: {}", scale);
        for tree in &mut all_trees {
            scale_node(&mut tree.root, scale);
        }
    }

    let base_score = if biases.len() == 1 { biases[0] } else { 0.0 };
    let base_scores = if biases.len() > 1 { biases } else { vec![] };

    Ok(Forest {
        trees: all_trees,
        base_score,
        base_scores,
        aggregation: AggregationKind::Sum,
        post_transform,
        catboost_metadata,
    })
}

fn scale_node(node: &mut Node, scale: f64) {
    match node {
        Node::Leaf { value } => *value *= scale,
        Node::Split {
            left_child,
            right_child,
            ..
        } => {
            scale_node(left_child, scale);
            scale_node(right_child, scale);
        }
    }
}

enum CatboostSplit {
    Numeric { feature_idx: usize, threshold: f32 },
    Ctr { ctr_idx: usize, threshold: f32 },
}

fn parse_oblivious_tree(tv: &Value, classes_count: usize) -> Result<Vec<Tree>> {
    let splits_val = tv
        .get("splits")
        .ok_or_else(|| anyhow!("Missing 'splits' in tree"))?;
    let splits_arr = splits_val
        .as_array()
        .ok_or_else(|| anyhow!("'splits' is not an array"))?;

    let mut splits = Vec::with_capacity(splits_arr.len());
    for sv in splits_arr {
        let split_type = sv["split_type"].as_str().unwrap_or("FloatFeature");
        if split_type == "FloatFeature" {
            let feature_idx = sv["float_feature_index"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing 'float_feature_index'"))?
                as usize;
            let threshold = sv["border"]
                .as_f64()
                .ok_or_else(|| anyhow!("Missing 'border'"))? as f32;
            splits.push(CatboostSplit::Numeric {
                feature_idx,
                threshold,
            });
        } else if split_type == "OnlineCtr" {
            let ctr_idx = sv["split_index"]
                .as_u64()
                .ok_or_else(|| anyhow!("Missing 'split_index' for OnlineCtr"))?
                as usize;
            let threshold = sv["border"]
                .as_f64()
                .ok_or_else(|| anyhow!("Missing 'border'"))? as f32;
            splits.push(CatboostSplit::Ctr { ctr_idx, threshold });
        } else {
            anyhow::bail!("Unsupported CatBoost split type: {}", split_type);
        }
    }

    let leaf_values_val = tv
        .get("leaf_values")
        .ok_or_else(|| anyhow!("Missing 'leaf_values' in tree"))?;
    let leaf_values = leaf_values_val
        .as_array()
        .ok_or_else(|| anyhow!("'leaf_values' is not an array"))?
        .iter()
        .map(|v| v.as_f64().ok_or_else(|| anyhow!("Invalid leaf value")))
        .collect::<Result<Vec<f64>>>()?;

    let n_leaves_per_class = 1 << splits.len();
    if leaf_values.len() != n_leaves_per_class * classes_count {
        anyhow::bail!(
            "Tree values mismatch: expected {} values ({} classes * {} leaves), got {}",
            n_leaves_per_class * classes_count,
            classes_count,
            n_leaves_per_class,
            leaf_values.len()
        );
    }

    // Multiclass leaf_values are leaf-major: [leaf0_class0, leaf0_class1, ...,
    // leaf1_class0, ...]. Verified against CatBoost's RawFormulaVal output.
    let mut trees = Vec::with_capacity(classes_count);
    for c in 0..classes_count {
        let class_leaf_values: Vec<f64> = (0..n_leaves_per_class)
            .map(|leaf| leaf_values[leaf * classes_count + c])
            .collect();
        let root = build_oblivious_node(&splits, &class_leaf_values);
        trees.push(Tree { root, weight: 1.0 });
    }

    Ok(trees)
}

fn build_oblivious_node(splits: &[CatboostSplit], leaf_values: &[f64]) -> Node {
    if splits.is_empty() {
        return Node::Leaf {
            value: leaf_values[0],
        };
    }

    let (current_split, remaining_splits) = splits.split_last().unwrap();
    let mid = leaf_values.len() / 2;
    let (right_half, left_half) = leaf_values.split_at(mid);

    let split_kind = match current_split {
        CatboostSplit::Numeric { threshold, .. } => SplitKind::Numeric {
            threshold: *threshold,
            operator: Operator::GreaterThan,
        },
        CatboostSplit::Ctr {
            ctr_idx, threshold, ..
        } => SplitKind::OnlineCtr {
            ctr_idx: *ctr_idx,
            threshold: *threshold,
        },
    };

    let feature_idx = match current_split {
        CatboostSplit::Numeric { feature_idx, .. } => *feature_idx,
        CatboostSplit::Ctr { .. } => 0,
    };

    Node::Split {
        feature_idx,
        split: split_kind,
        left_child: Box::new(build_oblivious_node(remaining_splits, left_half)),
        right_child: Box::new(build_oblivious_node(remaining_splits, right_half)),
        missing_direction: MissingDirection::None,
    }
}
