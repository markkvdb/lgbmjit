//! A plain interpreter with LightGBM's exact prediction rules, used to check the compiled code.
use crate::model::{Forest, MissingType, Tree};

/// LightGBM's `kZeroThreshold`: `1e-35f` widened to a double.
pub const ZERO_THRESHOLD: f64 = 1e-35f32 as f64;

/// Whether input `x` goes left at internal node `n` of `tree`.
pub fn go_left(tree: &Tree, n: usize, x: f64) -> bool {
    if tree.is_categorical(n) {
        if x.is_nan() { return false; }
        let v = x as i32;
        if v < 0 { return false; }
        let bits = tree.bitset(n);
        let word = (v / 32) as usize;
        return word < bits.len() && (bits[word] >> (v % 32)) & 1 == 1;
    }
    let mt = tree.missing_type(n);
    let x = if x.is_nan() && mt != MissingType::NaN { 0.0 } else { x };
    if (mt == MissingType::Zero && (-ZERO_THRESHOLD..=ZERO_THRESHOLD).contains(&x)) || (mt == MissingType::NaN && x.is_nan()) { return tree.default_left(n); }
    x <= tree.threshold[n]
}

/// Value of the leaf that `row` reaches in `tree`.
pub fn predict_tree(tree: &Tree, row: &[f64]) -> f64 {
    if tree.num_nodes() == 0 { return tree.leaf_value[0]; }
    let mut c = 0i32;
    while c >= 0 {
        let n = c as usize;
        c = if go_left(tree, n, row[tree.split_feature[n] as usize]) { tree.left[n] } else { tree.right[n] };
    }
    tree.leaf_value[!c as usize]
}

/// The boosting iterations a prediction uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IterRange { pub start: usize, pub count: usize }

impl IterRange {
    /// LightGBM's rule: `start` is clamped to the iterations the model has; `num_iteration <= 0` means all remaining.
    pub fn resolve(forest: &Forest, start_iteration: i64, num_iteration: i64) -> IterRange {
        let total = forest.num_iterations();
        let start = start_iteration.clamp(0, total as i64) as usize;
        let left = total - start;
        let count = if num_iteration > 0 { (num_iteration as usize).min(left) } else { left };
        IterRange { start, count }
    }
}

/// LightGBM's dense-row conversion drops inputs with `|x| <= kZeroThreshold`, so they reach the trees as 0.0.
pub fn canonical(x: f64) -> f64 { if x.abs() <= ZERO_THRESHOLD { 0.0 } else { x } }

/// Raw scores of one row: `out[k]` is the sum, in iteration order, of the class-`k` trees in `range`.
pub fn predict_row_raw(forest: &Forest, range: IterRange, row: &[f64], out: &mut [f64]) {
    let k = forest.trees_per_iteration;
    out.fill(0.0);
    let row: Vec<f64> = row.iter().map(|&x| canonical(x)).collect();
    for i in range.start..range.start + range.count { for (c, o) in out.iter_mut().enumerate() { *o += predict_tree(&forest.trees[i * k + c], &row); } }
}
