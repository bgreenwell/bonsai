use anyhow::Result;
use byteorder::{LittleEndian, ReadBytesExt};
use std::io::{Cursor, Read};

/// Represents a bitset used for categorical splits.
#[derive(Debug, Clone)]
pub struct BitSet {
    pub bitoff: u16,
    pub nbits: u32,
    pub data: Vec<u8>,
}

impl BitSet {
    pub fn contains(&self, idx: i32) -> bool {
        let idx = idx - self.bitoff as i32;
        if idx < 0 || idx >= self.nbits as i32 {
            return false;
        }
        let byte_idx = (idx >> 3) as usize;
        let bit_idx = (idx & 7) as u8;
        (self.data[byte_idx] & (1 << bit_idx)) != 0
    }
}

/// Represents the direction a missing value (NaN) should take.
#[derive(Debug, Clone, Copy)]
pub enum NaSplitDir {
    None = 0,
    NaVsRest = 1,
    NaLeft = 2,
    NaRight = 3,
    Left = 4,
    Right = 5,
}

impl NaSplitDir {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => NaSplitDir::NaVsRest,
            2 => NaSplitDir::NaLeft,
            3 => NaSplitDir::NaRight,
            4 => NaSplitDir::Left,
            5 => NaSplitDir::Right,
            _ => NaSplitDir::None,
        }
    }
}

/// Represents the type of split in an internal node.
#[derive(Debug)]
pub enum Split {
    Numeric {
        split_value: f32,
    },
    Categorical {
        bitset: BitSet,
    },
}

/// Represents a node in the decision tree.
#[derive(Debug)]
#[allow(dead_code)]
pub enum TreeNode {
    Internal {
        col_id: u16,
        na_split_dir: NaSplitDir,
        split: Split,
        left_child: Box<TreeNode>,
        right_child: Box<TreeNode>,
    },
    Leaf {
        prediction: f32,
    },
}

/// Statistics about a parsed tree
pub struct TreeStats {
    pub total_nodes: usize,
    pub leaf_nodes: usize,
    pub depth: usize,
}

/// Analyzes a tree to collect statistics
pub fn analyze_tree(tree: &TreeNode) -> TreeStats {
    fn analyze_recursive(node: &TreeNode, depth: usize) -> (usize, usize, usize) {
        match node {
            TreeNode::Leaf { .. } => (1, 1, depth),
            TreeNode::Internal { left_child, right_child, .. } => {
                let (left_nodes, left_leaves, left_depth) = analyze_recursive(left_child, depth + 1);
                let (right_nodes, right_leaves, right_depth) = analyze_recursive(right_child, depth + 1);
                (
                    1 + left_nodes + right_nodes,
                    left_leaves + right_leaves,
                    left_depth.max(right_depth),
                )
            }
        }
    }

    let (total_nodes, leaf_nodes, depth) = analyze_recursive(tree, 0);
    TreeStats {
        total_nodes,
        leaf_nodes,
        depth,
    }
}

/// Main parsing function for a tree.
///
/// Parses H2O MOJO tree format (version 1.40+).
///
/// Based on H2O's scoreTree implementation in SharedTreeMojoModel.java.
/// The tree is stored linearly with predictions embedded in the tree file.
///
/// Parameters
/// ----------
/// tree_data : &[u8]
///     Complete tree structure from .bin file (after 4-byte header)
/// _aux_data : &[u8]
///     Auxiliary data (not used for basic tree structure, only for advanced features)
///
/// Returns
/// -------
/// Result<TreeNode>
///     The parsed tree root node
pub fn parse_tree(tree_data: &[u8], _aux_data: &[u8]) -> Result<TreeNode> {
    let mut cursor = Cursor::new(tree_data);
    parse_node(&mut cursor)
}

/// Recursively parses a node from the tree byte stream.
///
/// H2O MOJO tree format (version 1.40):
/// - Each node starts with: node_type (1 byte) + col_id (2 bytes, little-endian)
/// - If col_id == 65535: leaf node, followed by prediction (4 bytes, f32)
/// - Otherwise: internal node, followed by:
///   - na_split_dir (1 byte)
///   - split_value (4 bytes, f32) for numeric splits
///   - Then recursively: left child, right child
///
/// The node_type byte encodes information about child types:
/// - lmask = node_type & 0x33: left child info
/// - rmask = (node_type & 0xC0) >> 2: right child info
/// - If (mask & 0x10) != 0: child is a leaf (stores 4-byte prediction)
/// - Otherwise: child is an internal node (recurse)
fn parse_node(cursor: &mut Cursor<&[u8]>) -> Result<TreeNode> {
    // Read node header
    let node_type = cursor.read_u8()?;
    let col_id = cursor.read_u16::<LittleEndian>()?;

    // Check if this is a leaf node
    if col_id == 65535 {
        let prediction = cursor.read_f32::<LittleEndian>()?;
        return Ok(TreeNode::Leaf { prediction });
    }

    // Internal node: read split information
    let na_split_dir_raw = cursor.read_u8()?;
    let na_split_dir = NaSplitDir::from_u8(na_split_dir_raw);

    // Check if this is a categorical split (bitset) or numeric split
    let equal = node_type & 0x0C;  // Bits indicating split type: 0, 8, or 12
    let na_vs_rest = matches!(na_split_dir, NaSplitDir::NaVsRest);

    let split = if !na_vs_rest && equal != 0 {
        // Categorical split: variable-length bitset
        if equal == 8 {
            // fill2: 4 bytes inline bitset (assumes 32 bits, bitoff=0)
            let mut data = vec![0u8; 4];
            cursor.read_exact(&mut data)?;
            Split::Categorical {
                bitset: BitSet {
                    bitoff: 0,
                    nbits: 32,
                    data,
                },
            }
        } else {
            // fill3: reads bitoff (u16) + nbits (u32), then reads bytes(nbits)
            let bitoff = cursor.read_u16::<LittleEndian>()?;
            let nbits = cursor.read_u32::<LittleEndian>()?;
            let nbytes = ((nbits.saturating_sub(1)) >> 3) + 1;
            let mut data = vec![0u8; nbytes as usize];
            cursor.read_exact(&mut data)?;
            Split::Categorical {
                bitset: BitSet {
                    bitoff,
                    nbits,
                    data,
                },
            }
        }
    } else if na_vs_rest {
        // FIX: NaVsRest splits solely on NaN-ness, no threshold value is stored.
        // We use NAN as a placeholder since the threshold is unused.
        Split::Numeric { split_value: f32::NAN }
    } else {
        // Numeric split: 4-byte float
        let split_value = cursor.read_f32::<LittleEndian>()?;
        Split::Numeric { split_value }
    };

    // Decode child type flags from node_type
    let lmask = node_type & 0x33;  // 0b00110011
    let rmask = (node_type & 0xC0) >> 2;  // 0b11000000 >> 2

    let left_is_leaf = (lmask & 0x10) != 0;
    let right_is_leaf = (rmask & 0x10) != 0;

    // IMPORTANT: If left child is NOT a leaf, there is a "subtree size" field
    // immediately following the split info. We must skip it to read the child itself.
    if !left_is_leaf {
        let skip_size = (lmask & 0x03) + 1; // 1 to 4 bytes
        cursor.set_position(cursor.position() + skip_size as u64);
    }

    // Parse left child
    let left_child = if left_is_leaf {
        let prediction = cursor.read_f32::<LittleEndian>()?;
        Box::new(TreeNode::Leaf { prediction })
    } else {
        Box::new(parse_node(cursor)?)
    };

    // Parse right child
    let right_child = if right_is_leaf {
        let prediction = cursor.read_f32::<LittleEndian>()?;
        Box::new(TreeNode::Leaf { prediction })
    } else {
        Box::new(parse_node(cursor)?)
    };

    Ok(TreeNode::Internal {
        col_id,
        na_split_dir,
        split,
        left_child,
        right_child,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitset_contains_basic() {
        // Bitset with bits 0, 1, 3 set: Binary 00001011 = 0x0B
        let bitset = BitSet {
            bitoff: 0,
            nbits: 4,
            data: vec![0x0B],
        };

        assert!(bitset.contains(0)); // Bit 0 set
        assert!(bitset.contains(1)); // Bit 1 set
        assert!(!bitset.contains(2)); // Bit 2 not set
        assert!(bitset.contains(3)); // Bit 3 set
    }

    #[test]
    fn test_bitset_contains_with_offset() {
        // Bitset starting at offset 5, bits 5-8
        let bitset = BitSet {
            bitoff: 5,
            nbits: 4,
            data: vec![0x0F], // All 4 bits set
        };

        assert!(!bitset.contains(4)); // Before offset
        assert!(bitset.contains(5)); // At offset
        assert!(bitset.contains(6));
        assert!(bitset.contains(7));
        assert!(bitset.contains(8));
        assert!(!bitset.contains(9)); // After nbits
    }

    #[test]
    fn test_bitset_contains_multibyte() {
        // Two-byte bitset: 0xFF 0x03 (10 bits total)
        let bitset = BitSet {
            bitoff: 0,
            nbits: 10,
            data: vec![0xFF, 0x03],
        };

        // First byte: all 8 bits set
        for i in 0..8 {
            assert!(bitset.contains(i));
        }

        // Second byte: only bits 8, 9 set
        assert!(bitset.contains(8));
        assert!(bitset.contains(9));
        assert!(!bitset.contains(10)); // Beyond nbits
    }

    #[test]
    fn test_bitset_contains_out_of_range() {
        let bitset = BitSet {
            bitoff: 5,
            nbits: 4,
            data: vec![0x0F],
        };

        // Test values outside the valid range
        assert!(!bitset.contains(-1));
        assert!(!bitset.contains(4)); // Just before offset
        assert!(!bitset.contains(100)); // Way after range
    }

    #[test]
    fn test_na_split_dir_from_u8() {
        assert!(matches!(NaSplitDir::from_u8(0), NaSplitDir::None));
        assert!(matches!(NaSplitDir::from_u8(1), NaSplitDir::NaVsRest));
        assert!(matches!(NaSplitDir::from_u8(2), NaSplitDir::NaLeft));
        assert!(matches!(NaSplitDir::from_u8(3), NaSplitDir::NaRight));
        assert!(matches!(NaSplitDir::from_u8(4), NaSplitDir::Left));
        assert!(matches!(NaSplitDir::from_u8(5), NaSplitDir::Right));
        assert!(matches!(NaSplitDir::from_u8(99), NaSplitDir::None)); // Unknown maps to None
    }

    #[test]
    fn test_parse_simple_leaf() {
        // Leaf node format: node_type (1 byte) + col_id=65535 (2 bytes) + prediction (4 bytes)
        let mut data = Vec::new();
        data.push(0x00); // node_type
        data.extend_from_slice(&65535u16.to_le_bytes()); // col_id = 65535 (leaf marker)
        data.extend_from_slice(&1.5f32.to_le_bytes()); // prediction

        let tree = parse_tree(&data, &[]).expect("Failed to parse leaf");

        match tree {
            TreeNode::Leaf { prediction } => {
                assert_eq!(prediction, 1.5);
            }
            _ => panic!("Expected leaf node"),
        }
    }

    #[test]
    fn test_parse_numeric_split_with_leaf_children() {
        // Internal node with numeric split and two leaf children
        let mut data = Vec::new();

        // Root node: node_type indicates both children are leaves
        data.push(0x51); // node_type: 0x51 = both children are leaves
        data.extend_from_slice(&0u16.to_le_bytes()); // col_id = 0 (feature 0)
        data.push(5); // na_split_dir = Right
        data.extend_from_slice(&10.0f32.to_le_bytes()); // split_value = 10.0

        // Left child (leaf)
        data.extend_from_slice(&1.0f32.to_le_bytes()); // prediction = 1.0

        // Right child (leaf)
        data.extend_from_slice(&(-1.0f32).to_le_bytes()); // prediction = -1.0

        let tree = parse_tree(&data, &[]).expect("Failed to parse tree");

        match tree {
            TreeNode::Internal {
                col_id,
                na_split_dir,
                split,
                left_child,
                right_child,
            } => {
                assert_eq!(col_id, 0);
                assert!(matches!(na_split_dir, NaSplitDir::Right));

                match split {
                    Split::Numeric { split_value } => {
                        assert_eq!(split_value, 10.0);
                    }
                    _ => panic!("Expected numeric split"),
                }

                match *left_child {
                    TreeNode::Leaf { prediction } => assert_eq!(prediction, 1.0),
                    _ => panic!("Expected left leaf"),
                }

                match *right_child {
                    TreeNode::Leaf { prediction } => assert_eq!(prediction, -1.0),
                    _ => panic!("Expected right leaf"),
                }
            }
            _ => panic!("Expected internal node"),
        }
    }

    #[test]
    fn test_parse_na_vs_rest_split() {
        // CRITICAL REGRESSION TEST (Feb 2026 Fix):
        // NaVsRest splits DON'T store a threshold value in MOJO format.
        // Reading 4 bytes here caused "failed to fill whole buffer" errors.
        // The fix: skip reading split_value and use f32::NAN as placeholder.
        let mut data = Vec::new();

        data.push(0x51); // node_type: both children are leaves
        data.extend_from_slice(&2u16.to_le_bytes()); // col_id = 2
        data.push(1); // na_split_dir = NaVsRest
        // NOTE: No split_value for NaVsRest!

        // Left child (leaf) - for non-NaN values
        data.extend_from_slice(&0.0f32.to_le_bytes());

        // Right child (leaf) - for NaN values
        data.extend_from_slice(&1.0f32.to_le_bytes());

        let tree = parse_tree(&data, &[]).expect("Failed to parse NaVsRest tree");

        match tree {
            TreeNode::Internal {
                col_id,
                na_split_dir,
                split,
                ..
            } => {
                assert_eq!(col_id, 2);
                assert!(matches!(na_split_dir, NaSplitDir::NaVsRest));

                match split {
                    Split::Numeric { split_value } => {
                        // Placeholder NAN value
                        assert!(split_value.is_nan());
                    }
                    _ => panic!("Expected numeric split with NAN"),
                }
            }
            _ => panic!("Expected internal node"),
        }
    }

    #[test]
    fn test_parse_categorical_split_fill2() {
        // Categorical split with inline 4-byte bitset (equal=8)
        let mut data = Vec::new();

        data.push(0x59); // node_type: 0x08 bit set (equal=8), both children leaves
        data.extend_from_slice(&1u16.to_le_bytes()); // col_id = 1
        data.push(5); // na_split_dir = Right
        data.extend_from_slice(&[0x0B, 0x00, 0x00, 0x00]); // 4-byte bitset: 0x0B = bits 0,1,3

        // Left child (leaf)
        data.extend_from_slice(&100.0f32.to_le_bytes());

        // Right child (leaf)
        data.extend_from_slice(&200.0f32.to_le_bytes());

        let tree = parse_tree(&data, &[]).expect("Failed to parse categorical tree");

        match tree {
            TreeNode::Internal { col_id, split, .. } => {
                assert_eq!(col_id, 1);

                match split {
                    Split::Categorical { bitset } => {
                        assert_eq!(bitset.bitoff, 0);
                        assert_eq!(bitset.nbits, 32);
                        assert_eq!(bitset.data.len(), 4);
                        assert_eq!(bitset.data[0], 0x0B);

                        // Verify bitset logic
                        assert!(bitset.contains(0));
                        assert!(bitset.contains(1));
                        assert!(!bitset.contains(2));
                        assert!(bitset.contains(3));
                    }
                    _ => panic!("Expected categorical split"),
                }
            }
            _ => panic!("Expected internal node"),
        }
    }

    #[test]
    fn test_parse_categorical_split_fill3() {
        // Categorical split with variable-length bitset (equal=12)
        let mut data = Vec::new();

        data.push(0x5D); // node_type: 0x0C bit set (equal=12), both children leaves
        data.extend_from_slice(&3u16.to_le_bytes()); // col_id = 3
        data.push(5); // na_split_dir = Right

        // Variable-length bitset
        data.extend_from_slice(&5u16.to_le_bytes()); // bitoff = 5
        data.extend_from_slice(&10u32.to_le_bytes()); // nbits = 10
        data.extend_from_slice(&[0xFF, 0x03]); // 2 bytes for 10 bits

        // Left child (leaf)
        data.extend_from_slice(&50.0f32.to_le_bytes());

        // Right child (leaf)
        data.extend_from_slice(&75.0f32.to_le_bytes());

        let tree = parse_tree(&data, &[]).expect("Failed to parse categorical fill3 tree");

        match tree {
            TreeNode::Internal { col_id, split, .. } => {
                assert_eq!(col_id, 3);

                match split {
                    Split::Categorical { bitset } => {
                        assert_eq!(bitset.bitoff, 5);
                        assert_eq!(bitset.nbits, 10);
                        assert_eq!(bitset.data.len(), 2);
                        assert_eq!(bitset.data[0], 0xFF);
                        assert_eq!(bitset.data[1], 0x03);

                        // Verify offset and range
                        assert!(!bitset.contains(4)); // Before offset
                        assert!(bitset.contains(5)); // At offset
                        assert!(bitset.contains(14)); // Last bit
                        assert!(!bitset.contains(15)); // Beyond nbits
                    }
                    _ => panic!("Expected categorical split"),
                }
            }
            _ => panic!("Expected internal node"),
        }
    }

    #[test]
    fn test_analyze_tree_simple_leaf() {
        let tree = TreeNode::Leaf { prediction: 1.0 };
        let stats = analyze_tree(&tree);

        assert_eq!(stats.total_nodes, 1);
        assert_eq!(stats.leaf_nodes, 1);
        assert_eq!(stats.depth, 0);
    }

    #[test]
    fn test_analyze_tree_with_split() {
        let tree = TreeNode::Internal {
            col_id: 0,
            na_split_dir: NaSplitDir::Right,
            split: Split::Numeric { split_value: 5.0 },
            left_child: Box::new(TreeNode::Leaf { prediction: 1.0 }),
            right_child: Box::new(TreeNode::Leaf { prediction: 2.0 }),
        };

        let stats = analyze_tree(&tree);

        assert_eq!(stats.total_nodes, 3); // 1 internal + 2 leaves
        assert_eq!(stats.leaf_nodes, 2);
        assert_eq!(stats.depth, 1);
    }

    #[test]
    fn test_analyze_tree_deeper() {
        // Create a tree with depth 2
        let tree = TreeNode::Internal {
            col_id: 0,
            na_split_dir: NaSplitDir::Right,
            split: Split::Numeric { split_value: 10.0 },
            left_child: Box::new(TreeNode::Internal {
                col_id: 1,
                na_split_dir: NaSplitDir::Left,
                split: Split::Numeric { split_value: 5.0 },
                left_child: Box::new(TreeNode::Leaf { prediction: 1.0 }),
                right_child: Box::new(TreeNode::Leaf { prediction: 2.0 }),
            }),
            right_child: Box::new(TreeNode::Leaf { prediction: 3.0 }),
        };

        let stats = analyze_tree(&tree);

        assert_eq!(stats.total_nodes, 5); // 2 internal + 3 leaves
        assert_eq!(stats.leaf_nodes, 3);
        assert_eq!(stats.depth, 2);
    }

    #[test]
    fn test_bitset_edge_case_large_offset() {
        let bitset = BitSet {
            bitoff: 1000,
            nbits: 10,
            data: vec![0xFF, 0x03],
        };

        assert!(!bitset.contains(999));
        assert!(bitset.contains(1000));
        assert!(bitset.contains(1009));
        assert!(!bitset.contains(1010));
    }

    #[test]
    fn test_bitset_bit_indexing() {
        // Test specific bit patterns to verify bit indexing logic
        let bitset = BitSet {
            bitoff: 0,
            nbits: 16,
            data: vec![0xAA, 0x55], // 10101010, 01010101
        };

        // First byte: 0xAA = 10101010 (bits 1,3,5,7 set)
        assert!(!bitset.contains(0));
        assert!(bitset.contains(1));
        assert!(!bitset.contains(2));
        assert!(bitset.contains(3));
        assert!(!bitset.contains(4));
        assert!(bitset.contains(5));
        assert!(!bitset.contains(6));
        assert!(bitset.contains(7));

        // Second byte: 0x55 = 01010101 (bits 0,2,4,6 set relative to byte)
        // Which are bits 8,10,12,14 absolute
        assert!(bitset.contains(8));
        assert!(!bitset.contains(9));
        assert!(bitset.contains(10));
        assert!(!bitset.contains(11));
        assert!(bitset.contains(12));
        assert!(!bitset.contains(13));
        assert!(bitset.contains(14));
        assert!(!bitset.contains(15));
    }

    #[test]
    fn test_parse_nested_internal_nodes() {
        // Test parsing with nested internal nodes to verify subtree size skipping
        // Root node with left child = internal node, right child = leaf
        let mut data = Vec::new();

        // Root node: node_type indicates left=internal, right=leaf
        data.push(0x41); // 0x41: left is internal (0x01), right is leaf (0x40)
        data.extend_from_slice(&0u16.to_le_bytes()); // col_id = 0
        data.push(5); // na_split_dir = Right
        data.extend_from_slice(&10.0f32.to_le_bytes()); // split_value = 10.0

        // Subtree size field for left child (internal node)
        // lmask = 0x41 & 0x33 = 0x01, so skip_size = (0x01 & 0x03) + 1 = 2 bytes
        data.extend_from_slice(&0x0000u16.to_le_bytes()); // 2-byte subtree size (value doesn't matter for test)

        // Left child (internal node)
        data.push(0x51); // Both children are leaves
        data.extend_from_slice(&1u16.to_le_bytes()); // col_id = 1
        data.push(4); // na_split_dir = Left
        data.extend_from_slice(&5.0f32.to_le_bytes()); // split_value = 5.0
        data.extend_from_slice(&1.0f32.to_le_bytes()); // Left grandchild (leaf)
        data.extend_from_slice(&2.0f32.to_le_bytes()); // Right grandchild (leaf)

        // Right child (leaf)
        data.extend_from_slice(&3.0f32.to_le_bytes());

        let tree = parse_tree(&data, &[]).expect("Failed to parse nested tree");

        match tree {
            TreeNode::Internal {
                col_id,
                left_child,
                right_child,
                ..
            } => {
                assert_eq!(col_id, 0);

                // Verify left child is internal
                match &*left_child {
                    TreeNode::Internal { col_id, .. } => {
                        assert_eq!(*col_id, 1);
                    }
                    _ => panic!("Expected internal left child"),
                }

                // Verify right child is leaf
                match *right_child {
                    TreeNode::Leaf { prediction } => {
                        assert_eq!(prediction, 3.0);
                    }
                    _ => panic!("Expected leaf right child"),
                }
            }
            _ => panic!("Expected internal root node"),
        }
    }

    #[test]
    fn test_parse_truncated_buffer() {
        // Test error handling for malformed/truncated binary data
        let mut data = Vec::new();

        // Start a node but don't provide enough bytes
        data.push(0x00); // node_type
        data.push(0x00); // Only 1 byte of col_id (need 2)

        let result = parse_tree(&data, &[]);
        assert!(result.is_err(), "Should fail on truncated buffer");
    }

    #[test]
    fn test_parse_empty_buffer() {
        // Test error handling for completely empty buffer
        let data = Vec::new();
        let result = parse_tree(&data, &[]);
        assert!(result.is_err(), "Should fail on empty buffer");
    }
}
