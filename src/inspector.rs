use crate::ir::*;
use std::collections::HashMap;

pub fn inspect(forest: &Forest, show_trees: bool, num_trees: usize) {
    print_header("MODEL METADATA");
    print_metadata(forest);

    print_header("TREE STATISTICS");
    print_tree_statistics(forest);

    print_header("FEATURE ANALYSIS");
    print_feature_analysis(forest);

    let has_categoricals = forest_has_categoricals(forest);
    if has_categoricals {
        print_header("CATEGORICAL FEATURES");
        print_categorical_details(forest);
    }

    if show_trees {
        print_header(&format!(
            "TREE STRUCTURES (showing {} trees)",
            num_trees.min(forest.trees.len())
        ));
        print_tree_structures(forest, num_trees);
    }

    print_validation(forest);
}

fn print_header(title: &str) {
    println!("\n{}", "=".repeat(70));
    println!("{}", title);
    println!("{}", "=".repeat(70));
}

fn print_metadata(forest: &Forest) {
    println!("Number of trees:    {}", forest.trees.len());
    println!("Aggregation:        {:?}", forest.aggregation);
    println!("Post-transform:     {:?}", forest.post_transform);
    println!("Base score:         {:.6}", forest.base_score);

    // Determine task type from post-transform
    let task = match &forest.post_transform {
        PostTransform::Logit => "Binary Classification".to_string(),
        PostTransform::Identity => "Regression".to_string(),
        PostTransform::Log => "Regression (log-transformed)".to_string(),
        PostTransform::Softmax { n_classes } => {
            format!("Multi-class Classification ({} classes)", n_classes)
        }
    };
    println!("Inferred task:      {}", task);
}

fn print_tree_statistics(forest: &Forest) {
    let mut depths = Vec::new();
    let mut node_counts = Vec::new();
    let mut leaf_counts = Vec::new();
    let mut numeric_splits = 0;
    let mut categorical_splits = 0;
    let mut oblivious_trees = 0;
    let mut missing_directions = HashMap::new();

    for tree in &forest.trees {
        if tree.root.get_oblivious_splits().is_some() {
            oblivious_trees += 1;
        }
        let (depth, nodes, leaves, numeric, categorical, missing) = analyze_tree(&tree.root);
        depths.push(depth);
        node_counts.push(nodes);
        leaf_counts.push(leaves);
        numeric_splits += numeric;
        categorical_splits += categorical;

        for (dir, count) in missing {
            *missing_directions.entry(dir).or_insert(0) += count;
        }
    }

    // Depth statistics
    let min_depth = depths.iter().min().copied().unwrap_or(0);
    let max_depth = depths.iter().max().copied().unwrap_or(0);
    let avg_depth = depths.iter().sum::<usize>() as f64 / depths.len().max(1) as f64;

    println!(
        "Tree depths:        min={}, max={}, avg={:.1}",
        min_depth, max_depth, avg_depth
    );

    if oblivious_trees > 0 {
        println!(
            "Oblivious trees:    {} / {} ({:.1}%)",
            oblivious_trees,
            forest.trees.len(),
            (oblivious_trees as f64 / forest.trees.len() as f64) * 100.0
        );
    }

    // Node statistics
    let total_nodes: usize = node_counts.iter().sum();
    let total_leaves: usize = leaf_counts.iter().sum();
    let avg_nodes = total_nodes as f64 / forest.trees.len() as f64;
    let avg_leaves = total_leaves as f64 / forest.trees.len() as f64;

    println!(
        "Nodes per tree:     avg={:.1} (total: {})",
        avg_nodes, total_nodes
    );
    println!(
        "Leaves per tree:    avg={:.1} (total: {})",
        avg_leaves, total_leaves
    );

    // Split type distribution
    let total_splits = numeric_splits + categorical_splits;
    if total_splits > 0 {
        let numeric_pct = (numeric_splits as f64 / total_splits as f64) * 100.0;
        let categorical_pct = (categorical_splits as f64 / total_splits as f64) * 100.0;
        println!("\nSplit types:");
        println!(
            "  Numeric:          {} ({:.1}%)",
            numeric_splits, numeric_pct
        );
        println!(
            "  Categorical:      {} ({:.1}%)",
            categorical_splits, categorical_pct
        );
    }

    // Missing value handling
    if !missing_directions.is_empty() {
        println!("\nMissing value handling:");
        for (dir, count) in missing_directions.iter() {
            println!("  {:?}:            {}", dir, count);
        }
    }
}

fn analyze_tree(
    node: &Node,
) -> (
    usize,
    usize,
    usize,
    usize,
    usize,
    HashMap<MissingDirection, usize>,
) {
    match node {
        Node::Leaf { .. } => (0, 1, 1, 0, 0, HashMap::new()),
        Node::Split {
            split,
            left_child,
            right_child,
            missing_direction,
            ..
        } => {
            let (
                left_depth,
                left_nodes,
                left_leaves,
                left_numeric,
                left_categorical,
                mut left_missing,
            ) = analyze_tree(left_child);
            let (
                right_depth,
                right_nodes,
                right_leaves,
                right_numeric,
                right_categorical,
                right_missing,
            ) = analyze_tree(right_child);

            let depth = 1 + left_depth.max(right_depth);
            let nodes = 1 + left_nodes + right_nodes;
            let leaves = left_leaves + right_leaves;

            let (numeric, categorical) = match split {
                SplitKind::Numeric { .. } => (1, 0),
                SplitKind::Categorical { .. } | SplitKind::OnlineCtr { .. } => (0, 1),
            };

            // Merge missing direction counts
            for (dir, count) in right_missing {
                *left_missing.entry(dir).or_insert(0) += count;
            }
            *left_missing.entry(*missing_direction).or_insert(0) += 1;

            (
                depth,
                nodes,
                leaves,
                numeric + left_numeric + right_numeric,
                categorical + left_categorical + right_categorical,
                left_missing,
            )
        }
    }
}

fn print_feature_analysis(forest: &Forest) {
    let mut feature_usage = HashMap::new();
    let mut categorical_features = std::collections::HashSet::new();

    for tree in &forest.trees {
        collect_feature_usage(&tree.root, &mut feature_usage, &mut categorical_features);
    }

    if feature_usage.is_empty() {
        println!("No features found (all trees are single leaves)");
        return;
    }

    let max_feature_idx = *feature_usage.keys().max().unwrap();
    println!("Number of features: {}", max_feature_idx + 1);
    println!("Features used:      {}", feature_usage.len());

    // Sort by usage count
    let mut sorted_features: Vec<_> = feature_usage.iter().collect();
    sorted_features.sort_by(|a, b| b.1.cmp(a.1));

    println!("\nTop 10 most-used features:");
    for (idx, (feature_idx, count)) in sorted_features.iter().take(10).enumerate() {
        let feature_type = if categorical_features.contains(feature_idx) {
            "categorical"
        } else {
            "numeric"
        };
        println!(
            "  {}. Feature {:<4} ({:>12}): {} splits",
            idx + 1,
            feature_idx,
            feature_type,
            count
        );
    }

    if feature_usage.len() != max_feature_idx + 1 {
        let unused = (max_feature_idx + 1) - feature_usage.len();
        println!(
            "\n⚠ Warning: {} feature(s) are never used in splits",
            unused
        );
    }
}

fn collect_feature_usage(
    node: &Node,
    usage: &mut HashMap<usize, usize>,
    categorical: &mut std::collections::HashSet<usize>,
) {
    if let Node::Split {
        feature_idx,
        split,
        left_child,
        right_child,
        ..
    } = node
    {
        *usage.entry(*feature_idx).or_insert(0) += 1;

        if matches!(split, SplitKind::Categorical { .. }) {
            categorical.insert(*feature_idx);
        }

        collect_feature_usage(left_child, usage, categorical);
        collect_feature_usage(right_child, usage, categorical);
    }
}

fn forest_has_categoricals(forest: &Forest) -> bool {
    for tree in &forest.trees {
        if tree_has_categoricals(&tree.root) {
            return true;
        }
    }
    false
}

fn tree_has_categoricals(node: &Node) -> bool {
    match node {
        Node::Leaf { .. } => false,
        Node::Split {
            split,
            left_child,
            right_child,
            ..
        } => {
            matches!(split, SplitKind::Categorical { .. })
                || tree_has_categoricals(left_child)
                || tree_has_categoricals(right_child)
        }
    }
}

fn print_categorical_details(forest: &Forest) {
    let mut categorical_info: HashMap<usize, Vec<CategoricalSplitInfo>> = HashMap::new();

    for tree in &forest.trees {
        collect_categorical_info(&tree.root, &mut categorical_info);
    }

    if categorical_info.is_empty() {
        println!("No categorical features found");
        return;
    }

    let mut sorted_features: Vec<_> = categorical_info.keys().collect();
    sorted_features.sort();

    for feature_idx in sorted_features {
        let infos = &categorical_info[feature_idx];
        println!("\nFeature {}:", feature_idx);
        println!("  Number of categorical splits: {}", infos.len());

        // Analyze bitset ranges; unwraps can't fail - a feature only has a map
        // entry if at least one categorical split was collected for it
        let min_bitoff = infos.iter().map(|i| i.bitoff).min().unwrap();
        let max_bitoff = infos.iter().map(|i| i.bitoff).max().unwrap();
        let min_nbits = infos.iter().map(|i| i.nbits).min().unwrap();
        let max_nbits = infos.iter().map(|i| i.nbits).max().unwrap();

        println!("  Bitset offset range: {} to {}", min_bitoff, max_bitoff);
        println!(
            "  Bitset size range:   {} to {} levels",
            min_nbits, max_nbits
        );

        // Show first bitset as example
        if let Some(first) = infos.first() {
            println!(
                "  Example bitset:      bitoff={}, nbits={}, data_len={} bytes",
                first.bitoff, first.nbits, first.data_len
            );
            if first.data_len <= 8 {
                print!("    Data (hex):        ");
                for (i, byte) in first.data_sample.iter().take(first.data_len).enumerate() {
                    print!("{:02x}", byte);
                    if i < first.data_len - 1 {
                        print!(" ");
                    }
                }
                println!();
            }
        }
    }
}

#[derive(Clone)]
struct CategoricalSplitInfo {
    bitoff: u16,
    nbits: u32,
    data_len: usize,
    data_sample: Vec<u8>,
}

fn collect_categorical_info(node: &Node, info: &mut HashMap<usize, Vec<CategoricalSplitInfo>>) {
    match node {
        Node::Leaf { .. } => {}
        Node::Split {
            feature_idx,
            split,
            left_child,
            right_child,
            ..
        } => {
            if let SplitKind::Categorical {
                bitoff,
                nbits,
                data,
            } = split
            {
                info.entry(*feature_idx)
                    .or_default()
                    .push(CategoricalSplitInfo {
                        bitoff: *bitoff,
                        nbits: *nbits,
                        data_len: data.len(),
                        data_sample: data.iter().take(8).copied().collect(),
                    });
            }

            collect_categorical_info(left_child, info);
            collect_categorical_info(right_child, info);
        }
    }
}

fn print_tree_structures(forest: &Forest, num_trees: usize) {
    for (idx, tree) in forest.trees.iter().take(num_trees).enumerate() {
        println!("\n--- Tree {} (weight: {}) ---", idx, tree.weight);
        print_node(&tree.root, 0);
    }
}

fn print_node(node: &Node, depth: usize) {
    let indent = "  ".repeat(depth);

    match node {
        Node::Leaf { value } => {
            println!("{}└─ Leaf: {:.6}", indent, value);
        }
        Node::Split {
            feature_idx,
            split,
            left_child,
            right_child,
            missing_direction,
        } => {
            let split_desc = match split {
                SplitKind::Numeric {
                    threshold,
                    operator,
                } => {
                    format!("x[{}] {:?} {:.6}", feature_idx, operator, threshold)
                }
                SplitKind::Categorical {
                    bitoff,
                    nbits,
                    data,
                } => {
                    format!(
                        "x[{}] in bitset(off={}, bits={}, {} bytes)",
                        feature_idx,
                        bitoff,
                        nbits,
                        data.len()
                    )
                }
                SplitKind::OnlineCtr { ctr_idx, threshold } => {
                    format!("ctr[{}] > {:.6}", ctr_idx, threshold)
                }
            };

            let missing_desc = match missing_direction {
                MissingDirection::None => String::new(),
                _ => format!(" [missing: {:?}]", missing_direction),
            };

            println!("{}├─ Split: {}{}", indent, split_desc, missing_desc);
            print_node(left_child, depth + 1);
            print_node(right_child, depth + 1);
        }
    }
}

fn print_validation(forest: &Forest) {
    let mut warnings = Vec::new();

    // Check for empty trees
    if forest.trees.is_empty() {
        warnings.push("⚠ Forest has no trees!".to_string());
    }

    // Check for trees with only one leaf
    let single_leaf_trees = forest
        .trees
        .iter()
        .filter(|t| matches!(t.root, Node::Leaf { .. }))
        .count();

    if single_leaf_trees > 0 {
        warnings.push(format!(
            "⚠ {} tree(s) have only a single leaf node",
            single_leaf_trees
        ));
    }

    // Check for unusual base scores
    if forest.base_score.abs() > 1000.0 {
        warnings.push(format!(
            "⚠ Unusually large base score: {:.2}",
            forest.base_score
        ));
    }

    if warnings.is_empty() {
        println!("\n✓ No validation issues detected");
    } else {
        print_header("VALIDATION WARNINGS");
        for warning in warnings {
            println!("{}", warning);
        }
    }
}
