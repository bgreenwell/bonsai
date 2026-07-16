use crate::ir::{
    AggregationKind, CatboostMetadata, Forest, MissingDirection, Node, Operator, PostTransform,
    SplitKind,
};
use anyhow::Result;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

/// Code layout strategy for the generated Rust source.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Layout {
    /// Pick automatically: array layout for large numeric-only forests,
    /// if/else otherwise.
    #[default]
    Auto,
    /// One nested if/else function per tree. Fastest at small model sizes;
    /// generated source grows with every node.
    IfElse,
    /// Flattened static arrays walked by a loop. Keeps rustc compile time
    /// practical for very large forests. Numeric splits only.
    Array,
}

/// Node count above which `Layout::Auto` switches to the array layout.
/// Below this, nested if/else compiles quickly and benchmarks faster.
pub const ARRAY_LAYOUT_NODE_THRESHOLD: usize = 10_000;

/// Decide which concrete layout to use for this forest.
pub fn resolve_layout(forest: &Forest, requested: Layout) -> Result<Layout> {
    let array_ok =
        forest.catboost_metadata.is_none() && forest.trees.iter().all(|t| numeric_only(&t.root));

    match requested {
        Layout::IfElse => Ok(Layout::IfElse),
        Layout::Array => {
            anyhow::ensure!(
                array_ok,
                "Array layout supports numeric splits only; this model contains \
                 categorical splits or CatBoost CTR features. Use --layout auto or ifelse."
            );
            Ok(Layout::Array)
        }
        Layout::Auto => {
            // Oblivious (CatBoost-style) trees already compile to compact
            // leaf-table lookups, so the if/else path stays cheap for them.
            let all_oblivious = forest
                .trees
                .iter()
                .all(|t| t.root.get_oblivious_splits().is_some());
            let n_nodes: usize = forest.trees.iter().map(|t| count_nodes(&t.root)).sum();
            if array_ok && !all_oblivious && n_nodes > ARRAY_LAYOUT_NODE_THRESHOLD {
                Ok(Layout::Array)
            } else {
                Ok(Layout::IfElse)
            }
        }
    }
}

fn numeric_only(node: &Node) -> bool {
    match node {
        Node::Leaf { .. } => true,
        Node::Split {
            split,
            left_child,
            right_child,
            ..
        } => {
            matches!(split, SplitKind::Numeric { .. })
                && numeric_only(left_child)
                && numeric_only(right_child)
        }
    }
}

fn count_nodes(node: &Node) -> usize {
    match node {
        Node::Leaf { .. } => 1,
        Node::Split {
            left_child,
            right_child,
            ..
        } => 1 + count_nodes(left_child) + count_nodes(right_child),
    }
}

/// Generate a standalone Rust source file from the given forest, choosing
/// the layout automatically. See [`generate_with_layout`].
/// The binary resolves the layout itself to report it; this wrapper is the
/// stable entry point for tests and future library consumers.
#[allow(dead_code)]
pub fn generate(forest: &Forest) -> Result<String> {
    generate_with_layout(forest, Layout::Auto)
}

/// Generate a standalone Rust source file from the given forest using the
/// requested layout (resolving `Auto` per [`resolve_layout`]).
pub fn generate_with_layout(forest: &Forest, layout: Layout) -> Result<String> {
    if forest.trees.is_empty() {
        anyhow::bail!("Cannot generate code for an empty forest (0 trees).");
    }
    match resolve_layout(forest, layout)? {
        Layout::Array => super::rust_array::generate(forest),
        _ => generate_ifelse(forest),
    }
}

/// Generate the if/else layout:
///
/// The output is a single module containing:
/// - A `bitset_contains` helper (only if categorical splits are present)
/// - CityHash + CTR helpers (only if CatBoost categorical features are present)
/// - A `Model` unit struct
/// - `Model::predict(&self, features: &[f32]) -> f32`
/// - One `#[inline(always)]` tree function per tree in the forest
fn generate_ifelse(forest: &Forest) -> Result<String> {
    // --- Conditional: bitset helper ---
    let needs_bitset = forest.trees.iter().any(|t| has_categorical(&t.root));
    let bitset_helper = if needs_bitset {
        quote! {
            fn bitset_contains(bitoff: u16, nbits: u32, data: &[u8], idx: i32) -> bool {
                let idx = idx - bitoff as i32;
                if idx < 0 || idx >= nbits as i32 {
                    return false;
                }
                let byte_idx = (idx >> 3) as usize;
                let bit_idx = (idx & 7) as u8;
                (data[byte_idx] & (1 << bit_idx)) != 0
            }
        }
    } else {
        quote! {}
    };

    // --- Conditional: CatBoost CTR helpers (computed once, used by both paths below) ---
    let catboost_helpers = if let Some(meta) = &forest.catboost_metadata {
        build_catboost_helpers(meta)?
    } else {
        quote! {}
    };

    // --- Per-tree scoring functions ---
    let tree_fns: Vec<TokenStream> = forest
        .trees
        .iter()
        .enumerate()
        .map(|(i, tree)| {
            let fn_name = format_ident!("tree_{}", i);
            if let Some(ob_splits) = tree.root.get_oblivious_splits() {
                let mut leaves = Vec::new();
                tree.root.collect_leaves(&mut leaves);
                compile_oblivious_tree(&fn_name, &ob_splits, &leaves)
            } else {
                let body = compile_node(&tree.root);
                quote! {
                    #[inline(always)]
                    #[allow(unused_variables)]
                    fn #fn_name(features: &[f32], ctrs: &[f64]) -> f64 {
                        #body
                    }
                }
            }
        })
        .collect();

    // --- Softmax (multiclass) path ---
    if let PostTransform::Softmax { n_classes } = &forest.post_transform {
        let n_classes = *n_classes;
        anyhow::ensure!(
            n_classes >= 2,
            "Softmax requires at least 2 classes, got {}",
            n_classes
        );
        anyhow::ensure!(
            forest.trees.len().is_multiple_of(n_classes),
            "Tree count ({}) must be divisible by n_classes ({}) for softmax",
            forest.trees.len(),
            n_classes
        );

        let output = build_softmax_model(
            forest,
            n_classes,
            &bitset_helper,
            &catboost_helpers,
            &tree_fns,
        )?;
        return Ok(output.to_string());
    }

    // --- Scalar path ---
    let tree_calls: Vec<TokenStream> = forest
        .trees
        .iter()
        .enumerate()
        .map(|(i, tree)| {
            let fn_name = format_ident!("tree_{}", i);
            let w = tree.weight;
            quote! { Self::#fn_name(features, ctrs) * #w }
        })
        .collect();

    let aggregation_expr = match &forest.aggregation {
        AggregationKind::Sum => {
            let base = forest.base_score;
            quote! { #base + #(#tree_calls)+* }
        }
        AggregationKind::Average => {
            let n = forest.trees.len() as f64;
            quote! { (#(#tree_calls)+*) / #n }
        }
    };

    let post_transform_expr = post_transform_expr(&forest.post_transform);

    let (predict_cat_method, internal_predict_call) = if forest.catboost_metadata.is_some() {
        (
            quote! {
                pub fn predict_cat(&self, float_features: &[f32], cat_features: &[&str]) -> f32 {
                    let ctrs = calculate_ctrs(float_features, cat_features);
                    self.predict_internal(float_features, &ctrs)
                }
            },
            quote! { self.predict_internal(features, &[]) },
        )
    } else {
        (quote! {}, quote! { self.predict_internal(features, &[]) })
    };

    let output = quote! {
        /// Auto-generated by bonsai.
        /// Do not edit manually.
        #bitset_helper
        #[allow(unused_variables, dead_code, non_snake_case)]
        #catboost_helpers

        pub struct Model;

        #[allow(unused_variables, dead_code)]
        impl Model {
            pub const N_CLASSES: usize = 1;

            pub fn predict(&self, features: &[f32]) -> f32 {
                #internal_predict_call
            }

            #predict_cat_method

            pub fn predict_batch(&self, features: &[f32], n_features: usize, out: &mut [f32]) {
                for (i, row) in features.chunks_exact(n_features).enumerate() {
                    out[i] = self.predict(row);
                }
            }

            fn predict_internal(&self, features: &[f32], ctrs: &[f64]) -> f32 {
                let raw: f64 = #aggregation_expr;
                #post_transform_expr
            }

            #(#tree_fns)*
        }
    };

    Ok(output.to_string())
}

// ---------------------------------------------------------------------------
// CatBoost helpers codegen
// ---------------------------------------------------------------------------

/// Build the CityHash64 + CTR lookup helpers for CatBoost models.
/// Called at most once per `generate()` invocation.
fn build_catboost_helpers(meta: &CatboostMetadata) -> Result<TokenStream> {
    let cityhash_logic = quote! {
        const K0: u64 = 0xc3a5c85c97cb3127;
        const K1: u64 = 0xb492b66fbe98f273;
        const K2: u64 = 0x9ae16a3b2f90404f;
        const K3: u64 = 0xc949d7c7509e6557;

        #[inline(always)]
        fn rotate(val: u64, shift: u32) -> u64 {
            if shift == 0 { val } else { (val >> shift) | (val << (64 - shift)) }
        }

        #[inline(always)]
        fn shift_mix(val: u64) -> u64 { val ^ (val >> 47) }

        #[inline(always)]
        fn hash_len16(u: u64, v: u64) -> u64 {
            const K_MUL: u64 = 0x9ddfea08eb382d69;
            let mut a = (u ^ v).wrapping_mul(K_MUL);
            a ^= a >> 47;
            let mut b = (v ^ a).wrapping_mul(K_MUL);
            b ^= b >> 47;
            b.wrapping_mul(K_MUL)
        }

        fn fetch64(s: &[u8], off: usize) -> u64 {
            let mut b = [0u8; 8];
            b.copy_from_slice(&s[off..off + 8]);
            u64::from_le_bytes(b)
        }

        fn fetch32(s: &[u8], off: usize) -> u32 {
            let mut b = [0u8; 4];
            b.copy_from_slice(&s[off..off + 4]);
            u32::from_le_bytes(b)
        }

        fn city_hash64_len0to16(s: &[u8]) -> u64 {
            let len = s.len();
            if len > 8 {
                let a = fetch64(s, 0);
                let b = fetch64(s, len - 8);
                return hash_len16(a, rotate(b.wrapping_add(len as u64), len as u32).wrapping_add(b)) ^ b;
            }
            if len >= 4 {
                let a = fetch32(s, 0) as u64;
                let b = fetch32(s, len - 4) as u64;
                return hash_len16((len as u64).wrapping_add(a << 3), b);
            }
            if len > 0 {
                let a = s[0] as u32;
                let b = s[len >> 1] as u32;
                let c = s[len - 1] as u32;
                let y = a.wrapping_add(b << 8);
                let z = (len as u32).wrapping_add(c << 2);
                return shift_mix((y as u64).wrapping_mul(K2) ^ (z as u64).wrapping_mul(K3)).wrapping_mul(K2);
            }
            K2
        }

        fn city_hash64_len17to32(s: &[u8]) -> u64 {
            let len = s.len();
            let a = fetch64(s, 0).wrapping_mul(K1);
            let b = fetch64(s, 8);
            let c = fetch64(s, len - 8).wrapping_mul(K2);
            let d = fetch64(s, len - 16).wrapping_mul(K0);
            hash_len16(
                rotate(a.wrapping_sub(b), 43).wrapping_add(rotate(c, 30)).wrapping_add(d),
                a.wrapping_add(rotate(b ^ K3, 20)).wrapping_sub(c).wrapping_add(len as u64),
            )
        }

        fn city_hash64_len33to64(s: &[u8]) -> u64 {
            let len = s.len();
            let z = fetch64(s, 24);
            let mut a = fetch64(s, 0).wrapping_add(
                (len as u64).wrapping_add(fetch64(s, len - 16)).wrapping_mul(K0));
            let mut b = rotate(a.wrapping_add(z), 52);
            let mut c = rotate(a, 37);
            a = a.wrapping_add(fetch64(s, 8));
            c = c.wrapping_add(rotate(a, 7));
            a = a.wrapping_add(fetch64(s, 16));
            let vf = a.wrapping_add(z);
            let vs = b.wrapping_add(rotate(a, 31)).wrapping_add(c);
            a = fetch64(s, 16).wrapping_add(fetch64(s, 24));
            let z2 = fetch64(s, len - 8);
            a = rotate(a.wrapping_add(z2), 52);
            b = rotate(a, 37);
            a = a.wrapping_add(fetch64(s, len - 24));
            b = b.wrapping_add(rotate(a, 11));
            a = a.wrapping_add(fetch64(s, len - 32));
            let wf = a.wrapping_add(z2);
            let ws = b.wrapping_add(fetch64(s, len - 16));
            let r = shift_mix(
                vf.wrapping_add(ws).wrapping_mul(K2)
                    .wrapping_add(wf.wrapping_add(vs).wrapping_mul(K0)));
            shift_mix(r.wrapping_mul(K0).wrapping_add(vs)).wrapping_mul(K2)
        }

        fn weak_hash32(s: &[u8], off: usize, mut a: u64, mut b: u64) -> (u64, u64) {
            let w = fetch64(s, off);
            let x = fetch64(s, off + 8);
            let y = fetch64(s, off + 16);
            let z = fetch64(s, off + 24);
            a = a.wrapping_add(w);
            b = rotate(b.wrapping_add(a).wrapping_add(z), 21);
            let c = a;
            a = a.wrapping_add(x).wrapping_add(y);
            b = b.wrapping_add(rotate(a, 44));
            (a.wrapping_add(z), b.wrapping_add(c))
        }

        #[inline(always)]
        fn city_hash64(s: &[u8]) -> u64 {
            let len = s.len();
            if len <= 16 { return city_hash64_len0to16(s); }
            if len <= 32 { return city_hash64_len17to32(s); }
            if len <= 64 { return city_hash64_len33to64(s); }
            let mut x = fetch64(s, len - 40);
            let mut y = fetch64(s, len - 16).wrapping_add(fetch64(s, len - 56));
            let mut z = hash_len16(
                fetch64(s, len - 48).wrapping_add(len as u64),
                fetch64(s, len - 24));
            let mut v = weak_hash32(s, len - 64, len as u64, z);
            let mut w = weak_hash32(s, len - 32, y.wrapping_add(K1), x);
            x = x.wrapping_mul(K1).wrapping_add(fetch64(s, 0));
            let mut pos = 0usize;
            let mut tail = (len - 1) & !63usize;
            loop {
                x = rotate(
                    x.wrapping_add(y).wrapping_add(v.0).wrapping_add(fetch64(s, pos + 8)), 37)
                    .wrapping_mul(K1);
                y = rotate(y.wrapping_add(v.1).wrapping_add(fetch64(s, pos + 48)), 42)
                    .wrapping_mul(K1);
                x ^= w.1;
                y = y.wrapping_add(v.0).wrapping_add(fetch64(s, pos + 40));
                z = rotate(z.wrapping_add(w.0), 33).wrapping_mul(K1);
                v = weak_hash32(s, pos, v.1.wrapping_mul(K1), x.wrapping_add(w.0));
                w = weak_hash32(s, pos + 32, z.wrapping_add(w.1),
                    y.wrapping_add(fetch64(s, pos + 16)));
                let tmp = z; z = x; x = tmp;
                pos += 64;
                tail -= 64;
                if tail == 0 { break; }
            }
            hash_len16(
                hash_len16(v.0, w.0).wrapping_add(shift_mix(y).wrapping_mul(K1)).wrapping_add(z),
                hash_len16(v.1, w.1).wrapping_add(x))
        }

        #[inline(always)]
        fn catboost_calc_hash(a: u64, b: u64) -> u64 {
            const MAGIC_MULT: u64 = 0x4906ba494954cb65;
            MAGIC_MULT.wrapping_mul(a.wrapping_add(MAGIC_MULT.wrapping_mul(b)))
        }
    };

    let mut ctr_tables = Vec::new();
    let mut ctr_elem_cat_indices = Vec::new();

    for ctr in &meta.ctrs {
        let table = meta.ctr_data.get(&ctr.identifier).ok_or_else(|| {
            anyhow::anyhow!("Missing CTR data for identifier: {}", ctr.identifier)
        })?;
        let hash_map = &table.hash_map;
        let stride = table.hash_stride;
        let denom = table.counter_denominator;
        let p_num = ctr.prior_numerator;
        let p_denom = ctr.prior_denominator;
        let shift = ctr.shift;
        let scale = ctr.scale;

        let mut elem_cat_indices = Vec::new();
        for elem in &ctr.elements {
            if elem.combination_element == "cat_feature_value" {
                elem_cat_indices.push(elem.cat_feature_index);
            }
        }
        ctr_elem_cat_indices.push(elem_cat_indices);

        ctr_tables.push(quote! {
            {
                let hash_map: &[u64] = &[#(#hash_map),*];
                let stride = #stride;
                let mut low = 0;
                let mut high = hash_map.len() / stride;
                let mut found_idx = None;
                while low < high {
                    let mid = low + (high - low) / 2;
                    let h = hash_map[mid * stride];
                    if h < combined_hash {
                        low = mid + 1;
                    } else if h > combined_hash {
                        high = mid;
                    } else {
                        found_idx = Some(mid);
                        break;
                    }
                }

                let (count_in_class, total_count) = if let Some(idx) = found_idx {
                    let base = idx * stride;
                    if stride == 3 {
                        (hash_map[base + 2] as f64, (hash_map[base + 1] + hash_map[base + 2]) as f64)
                    } else {
                        (0.0, #denom as f64)
                    }
                } else {
                    (0.0, #denom as f64)
                };

                let ctr_val = (count_in_class + #p_num) / (total_count + #p_denom);
                (ctr_val + #shift) * #scale
            }
        });
    }

    let ctr_calls: Vec<TokenStream> = ctr_tables
        .iter()
        .zip(ctr_elem_cat_indices.iter())
        .map(|(table_code, elem_indices)| {
            quote! {
                {
                    let mut combined_hash = 0u64;
                    #(
                        {
                            let cat_idx = #elem_indices;
                            let cat_val = cat_features[cat_idx];
                            let h32 = city_hash64(cat_val.as_bytes()) as u32;
                            combined_hash = catboost_calc_hash(combined_hash, (h32 as i32) as i64 as u64);
                        }
                    )*
                    #table_code
                }
            }
        })
        .collect();

    Ok(quote! {
        #cityhash_logic

        #[allow(unused_variables, dead_code)]
        fn calculate_ctrs(float_features: &[f32], cat_features: &[&str]) -> Vec<f64> {
            vec![
                #(#ctr_calls),*
            ]
        }
    })
}

// ---------------------------------------------------------------------------
// Node compilation
// ---------------------------------------------------------------------------

/// Recursively compile an IR node into a token stream of nested if/else blocks.
/// Map a scalar post-transform to the expression applied to `raw: f64`.
/// Softmax is handled by its own model builder and never reaches here.
pub(crate) fn post_transform_expr(post_transform: &PostTransform) -> TokenStream {
    match post_transform {
        PostTransform::Identity => quote! { raw as f32 },
        PostTransform::Logit => {
            quote! { (1.0f64 / (1.0f64 + (-raw).exp())) as f32 }
        }
        PostTransform::Log => {
            // Clamp before cast: exp() can exceed f32::MAX for large raw scores.
            quote! { raw.exp().min(f32::MAX as f64) as f32 }
        }
        PostTransform::Softmax { .. } => unreachable!("softmax handled separately"),
    }
}

/// Emit an f32 literal token. `quote!` panics on non-finite floats, and
/// LightGBM thresholds can overflow to infinity when narrowed from f64.
pub(crate) fn f32_token(v: f32) -> TokenStream {
    if v.is_finite() {
        quote! { #v }
    } else if v.is_nan() {
        quote! { f32::NAN }
    } else if v > 0.0 {
        quote! { f32::INFINITY }
    } else {
        quote! { f32::NEG_INFINITY }
    }
}

/// Emit an f64 literal token; see [`f32_token`].
pub(crate) fn f64_token(v: f64) -> TokenStream {
    if v.is_finite() {
        quote! { #v }
    } else if v.is_nan() {
        quote! { f64::NAN }
    } else if v > 0.0 {
        quote! { f64::INFINITY }
    } else {
        quote! { f64::NEG_INFINITY }
    }
}

fn compile_node(node: &Node) -> TokenStream {
    match node {
        Node::Leaf { value } => f64_token(*value),

        Node::Split {
            feature_idx,
            split,
            left_child,
            right_child,
            missing_direction,
        } => {
            let fi = *feature_idx;
            let left_code = compile_node(left_child);
            let right_code = compile_node(right_child);
            let left_cond = compile_condition(split, *missing_direction);

            // Each depth lives in its own block so `val` bindings shadow cleanly.
            quote! {
                {
                    let val = features[#fi];
                    if #left_cond {
                        #left_code
                    } else {
                        #right_code
                    }
                }
            }
        }
    }
}

fn compile_condition(split: &SplitKind, missing_direction: MissingDirection) -> TokenStream {
    match (missing_direction, split) {
        // NaVsRest splits solely on NaN-ness; the threshold value is unused.
        (MissingDirection::NaVsRest, _) => {
            quote! { !val.is_nan() }
        }

        (
            _,
            SplitKind::Numeric {
                threshold,
                operator,
            },
        ) => {
            let th = f32_token(*threshold);
            let cmp = match operator {
                Operator::LessThan => quote! { val < #th },
                Operator::LessOrEqual => quote! { val <= #th },
                Operator::GreaterThan => quote! { val > #th },
                Operator::GreaterOrEqual => quote! { val >= #th },
                Operator::Equal => quote! { val == #th },
                Operator::NotEqual => quote! { val != #th },
            };

            match missing_direction {
                MissingDirection::Left => quote! { val.is_nan() || (#cmp) },
                _ => quote! { !val.is_nan() && (#cmp) },
            }
        }

        (
            _,
            SplitKind::Categorical {
                bitoff,
                nbits,
                data,
            },
        ) => {
            let bo = *bitoff;
            let nb = *nbits;
            let bytes = data.clone();

            let membership = quote! {
                bitset_contains(#bo, #nb, &[#(#bytes),*], val as i32)
            };

            // H2O MOJO: value IN bitset → RIGHT child, NOT IN → LEFT child.
            match missing_direction {
                MissingDirection::Left => quote! { val.is_nan() || !#membership },
                _ => quote! { !val.is_nan() && !#membership },
            }
        }

        (_, SplitKind::OnlineCtr { ctr_idx, threshold }) => {
            let th = *threshold;
            quote! { ctrs[#ctr_idx] >= (#th as f64) }
        }
    }
}

fn compile_oblivious_tree(
    fn_name: &proc_macro2::Ident,
    splits: &[(usize, SplitKind, MissingDirection)],
    leaves: &[f64],
) -> TokenStream {
    // Bit order must match `collect_leaves`, which stores leaves with the
    // root split as the most significant bit (right-to-left within a level).
    let n = splits.len();
    let mut index_expr = quote! { 0usize };
    for (i, (fi, split, md)) in splits.iter().enumerate() {
        let bit = 1usize << (n - 1 - i);
        let cond = compile_condition(split, *md);
        index_expr = quote! {
            #index_expr | if { let val = features[#fi]; #cond } { #bit } else { 0 }
        };
    }

    let n_leaves = leaves.len();
    let leaf_tokens: Vec<TokenStream> = leaves.iter().map(|v| f64_token(*v)).collect();
    quote! {
        #[inline(always)]
        #[allow(unused_variables)]
        fn #fn_name(features: &[f32], ctrs: &[f64]) -> f64 {
            let index = #index_expr;
            const LEAVES: [f64; #n_leaves] = [#(#leaf_tokens),*];
            LEAVES[index]
        }
    }
}

// ---------------------------------------------------------------------------
// Softmax (multiclass) codegen
// ---------------------------------------------------------------------------

fn build_softmax_model(
    forest: &Forest,
    n_classes: usize,
    bitset_helper: &TokenStream,
    catboost_helpers: &TokenStream,
    tree_fns: &[TokenStream],
) -> Result<TokenStream> {
    let raw_decls: Vec<TokenStream> = (0..n_classes)
        .map(|c| {
            let raw_var = format_ident!("raw_{}", c);
            let base: f64 = if c < forest.base_scores.len() {
                forest.base_scores[c]
            } else {
                forest.base_score
            };

            let terms: Vec<TokenStream> = forest
                .trees
                .iter()
                .enumerate()
                .filter(|(i, _)| i % n_classes == c)
                .map(|(i, tree)| {
                    let fn_name = format_ident!("tree_{}", i);
                    let w = tree.weight;
                    quote! { Self::#fn_name(features, ctrs) * #w }
                })
                .collect();

            match forest.aggregation {
                AggregationKind::Sum => {
                    quote! { let #raw_var: f64 = #base + #(#terms)+*; }
                }
                AggregationKind::Average => {
                    let n = terms.len() as f64;
                    quote! { let #raw_var: f64 = (#(#terms)+*) / #n; }
                }
            }
        })
        .collect();

    let raw_vars: Vec<_> = (0..n_classes).map(|c| format_ident!("raw_{}", c)).collect();
    let first = &raw_vars[0];
    let max_expr = raw_vars
        .iter()
        .skip(1)
        .fold(quote! { #first }, |acc, v| quote! { #acc.max(#v) });
    let max_decl = quote! { let max_raw: f64 = #max_expr; };

    let exp_vars: Vec<_> = (0..n_classes).map(|c| format_ident!("e_{}", c)).collect();
    let exp_decls: Vec<TokenStream> = raw_vars
        .iter()
        .zip(exp_vars.iter())
        .map(|(r, e)| quote! { let #e: f64 = (#r - max_raw).exp(); })
        .collect();

    let sum_decl = quote! { let sum_e: f64 = #(#exp_vars)+*; };

    let prob_items: Vec<TokenStream> = exp_vars
        .iter()
        .map(|e| quote! { (#e / sum_e) as f32 })
        .collect();

    let predict_cat_proba_method = if forest.catboost_metadata.is_some() {
        quote! {
            pub fn predict_cat_proba(&self, float_features: &[f32], cat_features: &[&str]) -> Vec<f32> {
                let ctrs = calculate_ctrs(float_features, cat_features);
                self.predict_proba_internal(float_features, &ctrs)
            }
        }
    } else {
        quote! {}
    };

    Ok(quote! {
        /// Auto-generated by bonsai.
        /// Do not edit manually.
        #bitset_helper
        #[allow(unused_variables, dead_code, non_snake_case)]
        #catboost_helpers

        pub struct Model;

        #[allow(unused_variables, dead_code)]
        impl Model {
            pub const N_CLASSES: usize = #n_classes;

            pub fn predict_proba(&self, features: &[f32]) -> Vec<f32> {
                self.predict_proba_internal(features, &[])
            }

            #predict_cat_proba_method

            pub fn predict_batch(&self, features: &[f32], n_features: usize, out: &mut [f32]) {
                let n_classes = #n_classes;
                for (i, row) in features.chunks_exact(n_features).enumerate() {
                    let probs = self.predict_proba(row);
                    for (j, prob) in probs.into_iter().enumerate() {
                        out[i * n_classes + j] = prob;
                    }
                }
            }

            fn predict_proba_internal(&self, features: &[f32], ctrs: &[f64]) -> Vec<f32> {
                #(#raw_decls)*
                #max_decl
                #(#exp_decls)*
                #sum_decl
                vec![#(#prob_items),*]
            }

            #(#tree_fns)*
        }
    })
}

/// Walk the tree to check if any categorical split is present.
fn has_categorical(node: &Node) -> bool {
    match node {
        Node::Leaf { .. } => false,
        Node::Split {
            split,
            left_child,
            right_child,
            ..
        } => {
            matches!(split, SplitKind::Categorical { .. })
                || has_categorical(left_child)
                || has_categorical(right_child)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::*;

    fn leaf(v: f64) -> Node {
        Node::Leaf { value: v }
    }

    fn simple_split(fi: usize, th: f32, left: Node, right: Node) -> Node {
        Node::Split {
            feature_idx: fi,
            split: SplitKind::Numeric {
                threshold: th,
                operator: Operator::LessThan,
            },
            left_child: Box::new(left),
            right_child: Box::new(right),
            missing_direction: MissingDirection::Right,
        }
    }

    fn make_tree(root: Node) -> Tree {
        Tree { root, weight: 1.0 }
    }

    #[test]
    fn test_generate_oblivious_tree() {
        let split_kind = SplitKind::Numeric {
            threshold: 0.5,
            operator: Operator::GreaterThan,
        };
        let root = Node::Split {
            feature_idx: 0,
            split: split_kind.clone(),
            left_child: Box::new(Node::Leaf { value: 1.0 }),
            right_child: Box::new(Node::Leaf { value: 0.0 }),
            missing_direction: MissingDirection::None,
        };
        let forest = Forest {
            trees: vec![Tree { root, weight: 1.0 }],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };
        let code = generate(&forest).unwrap();
        assert!(code.contains("let index = 0usize | if"));
        assert!(code.contains("const LEAVES : [f64 ; 2usize] = [0f64 , 1f64]"));
    }

    #[test]
    fn test_generate_regression_identity() {
        let forest = Forest {
            trees: vec![make_tree(simple_split(0, 0.5, leaf(1.0), leaf(-1.0)))],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };
        let code = generate(&forest).unwrap();
        assert!(code.contains("pub fn predict"));
        assert!(code.contains("pub fn predict_batch"));
        assert!(code.contains("raw as f32"));
        assert!(!code.contains("predict_proba"));
    }

    #[test]
    fn test_generate_binary_logit() {
        let forest = Forest {
            trees: vec![make_tree(leaf(0.5))],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Logit,
            catboost_metadata: None,
        };
        let code = generate(&forest).unwrap();
        assert!(code.contains("pub fn predict"));
        assert!(code.contains("pub fn predict_batch"));
        assert!(code.contains("exp"));
        assert!(!code.contains("predict_proba"));
    }

    #[test]
    fn test_generate_multiclass_softmax_3_classes() {
        // 6 trees, 3 classes: tree 0,3 → class 0; tree 1,4 → class 1; tree 2,5 → class 2
        let trees: Vec<Tree> = (0..6).map(|i| make_tree(leaf(i as f64 * 0.1))).collect();
        let forest = Forest {
            trees,
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Softmax { n_classes: 3 },
            catboost_metadata: None,
        };
        let code = generate(&forest).unwrap();

        assert!(
            code.contains("predict_proba"),
            "should generate predict_proba"
        );
        assert!(
            code.contains("predict_batch"),
            "should generate predict_batch"
        );
        assert!(
            !code.contains("pub fn predict("),
            "should NOT generate predict"
        );
        assert!(
            code.contains("Vec < f32 >"),
            "return type should be Vec<f32>"
        );
        assert!(
            code.contains("max_raw"),
            "should include numerical stability max"
        );
        assert!(code.contains("sum_e"), "should include softmax denominator");
        assert!(code.contains("raw_0"), "class 0 raw score");
        assert!(code.contains("raw_1"), "class 1 raw score");
        assert!(code.contains("raw_2"), "class 2 raw score");

        let max_raw_decl_idx = code.find("let max_raw").expect("max_raw decl not found");
        let after_max = &code[max_raw_decl_idx..];
        assert!(
            after_max.contains(". max"),
            "max_raw should use .max() chaining"
        );
        let eq_pos = after_max.find('=').expect("= not found in max_raw decl");
        let after_eq = after_max[eq_pos + 1..].trim_start();
        assert!(
            !after_eq.starts_with("raw_0 raw_1"),
            "max_raw must not expand all idents: got '{}'",
            &after_eq[..after_eq.len().min(40)]
        );
    }

    #[test]
    fn test_generate_softmax_rejects_odd_tree_count() {
        let trees: Vec<Tree> = (0..5).map(|_| make_tree(leaf(0.0))).collect();
        let forest = Forest {
            trees,
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Softmax { n_classes: 3 },
            catboost_metadata: None,
        };
        assert!(
            generate(&forest).is_err(),
            "should error on indivisible tree count"
        );
    }

    #[test]
    fn test_generate_no_bitset_without_categoricals() {
        let forest = Forest {
            trees: vec![make_tree(leaf(1.0))],
            base_score: 0.0,
            base_scores: vec![],
            aggregation: AggregationKind::Sum,
            post_transform: PostTransform::Identity,
            catboost_metadata: None,
        };
        let code = generate(&forest).unwrap();
        assert!(!code.contains("bitset_contains"));
    }
}
