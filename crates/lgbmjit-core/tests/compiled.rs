mod common;
use common::{Rng, load, model_names};
use lgbmjit_core::keys::Matrix;
use lgbmjit_core::model::Forest;
use lgbmjit_core::output::finish_rows;
use lgbmjit_core::reference::{IterRange, predict_row_raw};
use lgbmjit_core::runtime::{Compiled, Dtype, Input, predict};
use lgbmjit_core::settings::Settings;

/// Random rows that exercise thresholds, missing values, zeros and out-of-range categories.
fn random_rows(forest: &Forest, rows: usize, seed: u64) -> Vec<f64> {
    let nf = forest.num_feature;
    let mut thresholds: Vec<Vec<f64>> = vec![vec![]; nf];
    for t in &forest.trees {
        for n in 0..t.num_nodes() { if !t.is_categorical(n) { thresholds[t.split_feature[n] as usize].push(t.threshold[n]); } }
    }
    let mut rng = Rng(seed);
    let mut out = Vec::with_capacity(rows * nf);
    for _ in 0..rows {
        for (f, ts) in thresholds.iter().enumerate() {
            let p = rng.unit();
            let x = if p < 0.1 { f64::NAN }
            else if forest.categorical[f] {
                match rng.below(8) { 0 => -0.5, 1 => -3.0, 2 => 1e10, _ => rng.below(64) as f64 + rng.unit() * 0.9 }
            } else if p < 0.13 { [0.0, -0.0, 1e-36, -1e-36, 1e-35f32 as f64][rng.below(5)] }
            else if p < 0.5 && !ts.is_empty() { let t = ts[rng.below(ts.len())]; [t, t.next_up(), t.next_down()][rng.below(3)] }
            else if !ts.is_empty() {
                let (lo, hi) = ts.iter().fold((f64::MAX, f64::MIN), |(a, b), &t| (a.min(t), b.max(t)));
                lo - 1.0 + rng.unit() * (hi - lo + 2.0)
            } else { rng.unit() * 100.0 - 50.0 };
            out.push(x);
        }
    }
    out
}

fn reference(forest: &Forest, range: IterRange, x: &[f64]) -> Vec<f64> {
    let (nf, k) = (forest.num_feature, forest.trees_per_iteration);
    let mut out = vec![0.0; x.len() / nf * k];
    for (row, o) in x.chunks_exact(nf).zip(out.chunks_exact_mut(k)) { predict_row_raw(forest, range, row, o); }
    out
}

fn compiled_raw(forest: &Forest, dtype: Dtype, range: IterRange, s: &Settings, x: &[f64], threads: usize) -> Vec<f64> {
    let nf = forest.num_feature;
    let rows = x.len() / nf;
    let c = Compiled::compile(forest, dtype, range, s).unwrap();
    let mut out = vec![f64::NAN; rows * forest.trees_per_iteration];
    match dtype {
        Dtype::F64 => c.predict_raw(&Input::F64(Matrix::from_slice_c(x, rows, nf)), &mut out, threads),
        Dtype::F32 => {
            let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
            c.predict_raw(&Input::F32(Matrix::from_slice_c(&x32, rows, nf)), &mut out, threads)
        }
    }
    out
}

fn as_dtype(x: &[f64], dtype: Dtype) -> Vec<f64> { match dtype { Dtype::F64 => x.to_vec(), Dtype::F32 => x.iter().map(|&v| v as f32 as f64).collect() } }

fn assert_bits_eq(got: &[f64], want: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() { assert!(g.to_bits() == w.to_bits(), "{what}: output {i}: got {g}, want {w}"); }
}

#[test]
fn matches_reference_all_models() {
    for name in model_names() {
        let f = load(&name);
        let range = IterRange::resolve(&f, 0, 0);
        let big = f.trees.len() > 500;
        let rows = if big { 300 } else { 2_000 };
        let x = random_rows(&f, rows, 7);
        let mut grid = vec![];
        for block_trees in [1, 8, 34] {
            for branch_free_leaves in [0, 4] {
                for cold_fraction in [0.0, 0.01] { grid.push(Settings { block_trees, tile_rows: 7, branch_free_leaves, cold_fraction }); }
            }
        }
        if big { grid = vec![grid[0], grid[6], grid[11]]; }
        for dtype in [Dtype::F64, Dtype::F32] {
            let want = reference(&f, range, &as_dtype(&x, dtype));
            for s in &grid {
                let got = compiled_raw(&f, dtype, range, s, &x, 1);
                assert_bits_eq(&got, &want, &format!("{name} {dtype:?} {s:?}"));
            }
        }
    }
}

#[test]
fn tile_sizes() {
    let f = load("airline");
    let range = IterRange::resolve(&f, 0, 0);
    let x = random_rows(&f, 1_000, 8);
    let want = reference(&f, range, &x);
    for tile_rows in [1, 7, 512, 4096] {
        let s = Settings { tile_rows, ..Settings::default_for_host() };
        assert_bits_eq(&compiled_raw(&f, Dtype::F64, range, &s, &x, 1), &want, &format!("tile {tile_rows}"));
    }
}

#[test]
fn iteration_ranges() {
    let f = load("multiclass");
    let range = IterRange { start: 2, count: 3 };
    let x = random_rows(&f, 500, 9);
    let want = reference(&f, range, &x);
    assert_bits_eq(&compiled_raw(&f, Dtype::F64, range, &Settings::default_for_host(), &x, 1), &want, "range");
    let empty = IterRange { start: f.num_iterations(), count: 0 };
    assert!(compiled_raw(&f, Dtype::F64, empty, &Settings::default_for_host(), &x, 1).iter().all(|&v| v == 0.0));
}

#[test]
fn threads_and_odd_sizes() {
    let f = load("NYC_taxi");
    let range = IterRange::resolve(&f, 0, 0);
    let s = Settings { tile_rows: 64, ..Settings::default_for_host() };
    for rows in [0, 1, 1023, 1025] {
        let x = random_rows(&f, rows, 10);
        let want = reference(&f, range, &x);
        for threads in [1, 3, 64] {
            assert_bits_eq(&compiled_raw(&f, Dtype::F64, range, &s, &x, threads), &want, &format!("{rows} rows, {threads} threads"));
        }
    }
}

#[test]
fn single_leaf_trees() {
    let f = load("single_leaf");
    let range = IterRange::resolve(&f, 0, 0);
    let x = random_rows(&f, 10, 11);
    assert_bits_eq(&compiled_raw(&f, Dtype::F32, range, &Settings::default_for_host(), &x, 1), &[2.0; 10], "single leaf");
}

#[test]
fn predict_applies_output_transform() {
    let f = load("multiclass");
    let range = IterRange::resolve(&f, 0, 0);
    let x = random_rows(&f, 100, 12);
    let c = Compiled::compile(&f, Dtype::F64, range, &Settings::default_for_host()).unwrap();
    let input = Input::F64(Matrix::from_slice_c(&x, 100, f.num_feature));
    let mut want = reference(&f, range, &x);
    assert_bits_eq(&predict(&f, &c, &input, true, 2), &want, "raw");
    finish_rows(&f, range, false, &mut want);
    assert_bits_eq(&predict(&f, &c, &input, false, 2), &want, "transformed");
}

#[test]
fn rejects_wrong_input() {
    let f = load("NYC_taxi");
    let c = Compiled::compile(&f, Dtype::F32, IterRange::resolve(&f, 0, 0), &Settings::default_for_host()).unwrap();
    let x = vec![0.0f64; 10];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut out = vec![0.0; 2];
        c.predict_raw(&Input::F64(Matrix::from_slice_c(&x, 2, 5)), &mut out, 1)
    }));
    assert!(r.is_err(), "an f64 input on an f32 compilation must be rejected");
}

#[test]
fn compiled_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Compiled>();
}

#[test]
fn one_node_models_match_reference() {
    let zt = 1e-35f32 as f64;
    let mut xs = vec![f64::NAN, 0.0, -0.0, 1e-36, -1e-36, zt, -zt, zt.next_up(), -zt.next_up(), 1.0, -1.0, 0.5, -0.5, 95.0, 64.0, 96.0, 1e10, -1e10];
    xs.extend([0.25f64, -0.25, 2.0, 3.0].iter().flat_map(|&t| [t, t.next_up(), t.next_down()]));
    let mut models = vec![];
    for mt in 0..3u8 {
        for dl in [false, true] {
            for t in [0.25, -0.25, 0.0, 1e-36, zt] { models.push(common::one_node_text(t, (mt << 2) | ((dl as u8) << 1), None, "regression")); }
        }
    }
    let wide =
        common::one_node_text(0.0, 1, Some(0), "regression").replace("cat_boundaries=0 1\ncat_threshold=0", "cat_boundaries=0 3\ncat_threshold=1 0 2147483648");
    models.push(wide);
    models.push(common::one_node_text(0.0, 1, Some(1 << 3), "regression"));
    for text in &models {
        let f = Forest::parse(text).unwrap();
        let range = IterRange::resolve(&f, 0, 0);
        for dtype in [Dtype::F64, Dtype::F32] {
            let want = reference(&f, range, &as_dtype(&xs, dtype));
            for branch_free_leaves in [0, 2] {
                let s = Settings { branch_free_leaves, ..Settings::default_for_host() };
                assert_bits_eq(&compiled_raw(&f, dtype, range, &s, &xs, 1), &want, &format!("{dtype:?} {text}"));
            }
        }
    }
}

#[test]
fn set_tile_rows_keeps_results() {
    let f = load("airline");
    let range = IterRange::resolve(&f, 0, 0);
    let x = random_rows(&f, 700, 13);
    let c = Compiled::compile(&f, Dtype::F64, range, &Settings::default_for_host()).unwrap();
    let input = Input::F64(Matrix::from_slice_c(&x, 700, f.num_feature));
    let want = reference(&f, range, &x);
    for tile in [1, 3, 300, 5000] {
        c.set_tile_rows(tile);
        assert_eq!(c.tile_rows(), tile);
        let mut out = vec![0.0; 700];
        c.predict_raw(&input, &mut out, 2);
        assert_bits_eq(&out, &want, &format!("tile {tile}"));
    }
}

#[test]
fn small_batches_use_fewer_threads() {
    use lgbmjit_core::runtime::effective_threads;
    assert_eq!(effective_threads(96, 15), 1);
    assert_eq!(effective_threads(0, 4), 1);
    assert_eq!(effective_threads(300, 0), 1);
    assert_eq!(effective_threads(1024, 15), 4);
    assert_eq!(effective_threads(100_000, 4), 4);
}
