use crate::ir::{
    AggregationKind, Forest, MissingDirection, Node, Operator, PostTransform, SplitKind, Tree,
};
use crate::parsers::{ini, tree_parser};
use anyhow::{Context, Result};
use std::io::Read;
use std::path::Path;
use zip::ZipArchive;

pub struct MojoFrontend;

impl MojoFrontend {
    pub fn new() -> Self {
        Self
    }
}

impl super::Frontend for MojoFrontend {
    fn parse(&self, path: &Path) -> Result<Forest> {
        let file =
            std::fs::File::open(path).with_context(|| format!("Failed to open {:?}", path))?;
        let mut archive =
            ZipArchive::new(file).context("Failed to read as ZIP. Is this a valid MOJO?")?;

        // --- Read and parse model.ini ---
        let mut ini_content = String::new();
        archive
            .by_name("model.ini")
            .context("Could not find 'model.ini' in the MOJO archive")?
            .read_to_string(&mut ini_content)?;

        let metadata = ini::parse_model_ini(&ini_content)?;

        // Print metadata summary (preserves h2o-poet UX)
        println!("---------------------------------------------");
        println!("  Model Metadata");
        println!("---------------------------------------------");
        println!("  H2O Version:   {}", metadata.h2o_version);
        println!("  Algorithm:     {}", metadata.algorithm);
        println!("  Trees:         {}", metadata.n_trees);
        println!("  Features:      {}", metadata.n_features);
        println!("  Distribution:  {}", metadata.distribution);
        println!("  Link Function: {}", metadata.link_function);
        println!("  Init F:        {}", metadata.init_f);
        println!("---------------------------------------------");

        if metadata.algorithm != "gbm" && metadata.algorithm != "drf" {
            anyhow::bail!(
                "Unsupported algorithm '{}'. Only 'gbm' and 'drf' are supported.",
                metadata.algorithm
            );
        }

        // --- Parse all trees ---
        let mut trees = Vec::with_capacity(metadata.n_trees);

        for tree_idx in 0..metadata.n_trees {
            let tree_filename = format!("trees/t00_{:03}.bin", tree_idx);
            let aux_filename = format!("trees/t00_{:03}_aux.bin", tree_idx);

            let mut tree_buffer = Vec::new();
            archive
                .by_name(&tree_filename)
                .with_context(|| format!("Missing tree file: {}", tree_filename))?
                .read_to_end(&mut tree_buffer)?;

            let mut aux_buffer = Vec::new();
            archive
                .by_name(&aux_filename)
                .with_context(|| format!("Missing aux file: {}", aux_filename))?
                .read_to_end(&mut aux_buffer)?;

            let parsed = tree_parser::parse_tree(&tree_buffer, &aux_buffer)
                .with_context(|| format!("Failed to parse tree {}", tree_idx))?;

            trees.push(Tree {
                root: convert_node(&parsed),
                weight: 1.0,
            });
        }

        println!(
            "  Parsed {}/{} trees successfully.",
            trees.len(),
            metadata.n_trees
        );

        // --- Determine aggregation and post-transform from metadata ---
        let aggregation = match metadata.algorithm.as_str() {
            "gbm" => AggregationKind::Sum,
            "drf" => AggregationKind::Average,
            _ => unreachable!(), // validated above
        };

        let post_transform = match metadata.link_function.as_str() {
            "logit" => PostTransform::Logit,
            "log" => PostTransform::Log,
            _ => PostTransform::Identity,
        };

        Ok(Forest {
            trees,
            base_score: metadata.init_f,
            base_scores: vec![],
            aggregation,
            post_transform,
        })
    }
}

/// Recursively convert a parsed h2o-poet `TreeNode` into the unified `ir::Node`.
fn convert_node(node: &tree_parser::TreeNode) -> Node {
    match node {
        tree_parser::TreeNode::Leaf { prediction } => Node::Leaf {
            value: *prediction as f64,
        },

        tree_parser::TreeNode::Internal {
            col_id,
            na_split_dir,
            split,
            left_child,
            right_child,
        } => {
            let split_kind = match split {
                tree_parser::Split::Numeric { split_value } => SplitKind::Numeric {
                    threshold: *split_value,
                    operator: Operator::LessThan, // H2O convention: left = val < threshold
                },
                tree_parser::Split::Categorical { bitset } => SplitKind::Categorical {
                    bitoff: bitset.bitoff,
                    nbits: bitset.nbits,
                    data: bitset.data.clone(),
                },
            };

            let missing_direction = match na_split_dir {
                tree_parser::NaSplitDir::None => MissingDirection::None,
                tree_parser::NaSplitDir::NaVsRest => MissingDirection::NaVsRest,
                tree_parser::NaSplitDir::NaLeft => MissingDirection::Left,
                tree_parser::NaSplitDir::NaRight => MissingDirection::Right,
                tree_parser::NaSplitDir::Left => MissingDirection::Left,
                tree_parser::NaSplitDir::Right => MissingDirection::Right,
            };

            Node::Split {
                feature_idx: *col_id as usize,
                split: split_kind,
                left_child: Box::new(convert_node(left_child)),
                right_child: Box::new(convert_node(right_child)),
                missing_direction,
            }
        }
    }
}
