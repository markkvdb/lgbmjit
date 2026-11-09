mod common;
use common::{load, one_node_text};
use lgbmjit_core::model::Forest;
use lgbmjit_core::output::finish_rows;
use lgbmjit_core::reference::{IterRange, go_left, predict_row_raw};

const ZERO_THRESHOLD: f64 = 1e-35f32 as f64;

fn node(threshold: f64, missing_type: u8, default_left: bool) -> Forest {
    let dt = (missing_type << 2) | ((default_left as u8) << 1);
    Forest::parse(&one_node_text(threshold, dt, None, "regression")).unwrap()
}

#[test]
fn numerical_missing_types() {
    let nan = f64::NAN;
    let up = f64::next_up;
    // (missing type, default left, threshold, input, goes left)
    let cases = [
        (0, false, 0.5, nan, true), (0, true, 0.5, nan, true), (0, false, -0.5, nan, false), (0, false, 0.5, 0.5, true), (0, false, 0.5, up(0.5), false),
        (0, false, -0.5, 0.0, false), (1, true, 0.5, nan, true), (1, true, 0.5, 0.0, true), (1, true, 0.5, 0.6, false), (1, false, 0.5, nan, false),
        (1, false, 0.5, 0.0, false), (1, false, 0.5, 1e-36, false), (1, false, 0.5, -1e-36, false), (1, false, 0.5, ZERO_THRESHOLD, false),
        (1, false, 0.5, up(ZERO_THRESHOLD), true), (1, false, 0.5, 0.4, true), (1, true, -0.5, -1e-36, true), (2, true, 0.5, nan, true),
        (2, false, 0.5, nan, false), (2, false, 0.5, 0.0, true), (2, true, -0.5, 0.0, false),
    ];
    for (mt, dl, t, x, want) in cases {
        let f = node(t, mt, dl);
        assert_eq!(go_left(&f.trees[0], 0, x), want, "missing type {mt}, default left {dl}, t {t}, x {x}");
    }
}

#[test]
fn categorical_rule() {
    let bits_8_11 = (1u32 << 8) | (1 << 11);
    let f = Forest::parse(&one_node_text(0.0, 1, Some(bits_8_11), "regression")).unwrap();
    let t = &f.trees[0];
    for (x, want) in [(8.0, true), (11.9, true), (9.0, false), (f64::NAN, false), (-0.5, false), (-3.0, false), (1e10, false), (40.0, false)] {
        assert_eq!(go_left(t, 0, x), want, "x {x}");
    }
    let f = Forest::parse(&one_node_text(0.0, 1, Some(1), "regression")).unwrap();
    assert!(go_left(&f.trees[0], 0, -0.5), "-0.5 truncates to category 0");
    assert!(!go_left(&f.trees[0], 0, -1.0));
}

#[test]
fn iter_range_resolution() {
    let f = load("NYC_taxi");
    let r = |s, n| IterRange::resolve(&f, s, n);
    assert_eq!(r(0, 0), IterRange { start: 0, count: 100 });
    assert_eq!(r(30, 0), IterRange { start: 30, count: 70 });
    assert_eq!(r(30, 40), IterRange { start: 30, count: 40 });
    assert_eq!(r(80, 50), IterRange { start: 80, count: 20 });
    assert_eq!(r(-2, 0), IterRange { start: 0, count: 100 });
    assert_eq!(r(120, -1), IterRange { start: 100, count: 0 });
}

#[test]
fn raw_prediction_sums_leaves() {
    let f = node(0.5, 2, false);
    let mut out = [0.0];
    predict_row_raw(&f, IterRange::resolve(&f, 0, 0), &[0.2], &mut out);
    assert_eq!(out, [1.0]);
    predict_row_raw(&f, IterRange::resolve(&f, 0, 0), &[f64::NAN], &mut out);
    assert_eq!(out, [2.0]);
}

fn finish(objective: &str, raw: &[f64], raw_score: bool) -> Vec<f64> {
    let f = Forest::parse(&one_node_text(0.5, 0, None, objective)).unwrap();
    let mut out = raw.to_vec();
    finish_rows(&f, IterRange::resolve(&f, 0, 0), raw_score, &mut out);
    out
}

#[test]
fn transforms() {
    assert_eq!(finish("regression sqrt", &[-3.0, 0.0, 2.0], false), [-9.0, 0.0, 4.0]);
    assert_eq!(finish("regression", &[-3.0], false), [-3.0]);
    assert_eq!(finish("poisson", &[0.5], false), [0.5f64.exp()]);
    assert_eq!(finish("binary sigmoid:2", &[0.5], false), [1.0 / (1.0 + (-1.0f64).exp())]);
    assert_eq!(finish("binary sigmoid:2", &[0.5], true), [0.5]);
    assert_eq!(finish("cross_entropy", &[0.5], false), [1.0 / (1.0 + (-0.5f64).exp())]);
    assert_eq!(finish("cross_entropy_lambda", &[0.5], false), [0.5f64.exp().ln_1p()]);
}

#[test]
fn softmax_and_ova() {
    let text = common::model_text("multiclass");
    let k = Forest::parse(&text).unwrap().trees_per_iteration;
    assert_eq!(k, 3);
    let f = Forest::parse(&text).unwrap();
    let mut out = vec![1.0, 2.0, 3.0];
    finish_rows(&f, IterRange::resolve(&f, 0, 0), false, &mut out);
    let e = [(1.0f64 - 3.0).exp(), (2.0f64 - 3.0).exp(), 0.0f64.exp()];
    let s = 0.0 + e[0] + e[1] + e[2];
    assert_eq!(out, [e[0] / s, e[1] / s, e[2] / s]);
    let f = Forest::parse(&text.replace("objective=multiclass num_class:3", "objective=multiclassova num_class:3 sigmoid:1")).unwrap();
    let mut out = vec![1.0, 2.0, 3.0];
    finish_rows(&f, IterRange::resolve(&f, 0, 0), false, &mut out);
    assert_eq!(out, [1.0, 2.0, 3.0].map(|x: f64| 1.0 / (1.0 + (-x).exp())));
}

#[test]
fn average_output_divides_by_iterations_unless_raw() {
    let text = one_node_text(0.5, 0, None, "regression").replace("tree_sizes=0", "tree_sizes=0\naverage_output");
    let f = Forest::parse(&text).unwrap();
    assert!(f.average_output);
    let range = IterRange { start: 0, count: 4 };
    let mut out = vec![2.0];
    finish_rows(&f, range, false, &mut out);
    assert_eq!(out, [0.5]);
    let mut out = vec![2.0];
    finish_rows(&f, range, true, &mut out);
    assert_eq!(out, [2.0]);
}

#[test]
fn tiny_inputs_become_zero() {
    // LightGBM's dense-row conversion drops |x| <= kZeroThreshold, so those inputs reach the trees as 0.0.
    let f = node(0.0, 0, false);
    let range = IterRange::resolve(&f, 0, 0);
    let mut out = [0.0];
    for x in [1e-36, -1e-36, 1e-35f32 as f64, -(1e-35f32 as f64)] {
        predict_row_raw(&f, range, &[x], &mut out);
        assert_eq!(out, [1.0], "x {x}");
    }
    predict_row_raw(&f, range, &[(1e-35f32 as f64).next_up()], &mut out);
    assert_eq!(out, [2.0]);
}
