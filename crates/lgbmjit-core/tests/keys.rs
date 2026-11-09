mod common;
use common::Rng;
use lgbmjit_core::keys::{Matrix, f32_at_or_below, fill_keys32, fill_keys64, key32, key64};

fn random_f32(rng: &mut Rng) -> f32 { loop { let x = f32::from_bits(rng.next_u64() as u32); if !x.is_nan() { return x; } } }

fn random_f64(rng: &mut Rng) -> f64 { loop { let x = f64::from_bits(rng.next_u64()); if !x.is_nan() { return x; } } }

fn edges_f32() -> Vec<f32> {
    vec![
        f32::NEG_INFINITY, -f32::MAX, -1.0, -f32::MIN_POSITIVE, -f32::from_bits(1), -0.0, 0.0, f32::from_bits(1), f32::MIN_POSITIVE, 1.0, f32::MAX,
        f32::INFINITY,
    ]
}

#[test]
fn key_order_f32() {
    let mut rng = Rng(1);
    let mut xs = edges_f32();
    xs.extend((0..100_000).map(|_| random_f32(&mut rng)));
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in xs.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a == b { assert_eq!(key32(a), key32(b), "{a} {b}"); } else { assert!(key32(a) < key32(b), "{a} {b}"); }
    }
}

#[test]
fn key_order_f64() {
    let mut rng = Rng(2);
    let mut xs: Vec<f64> = edges_f32().into_iter().map(f64::from).collect();
    xs.extend([-f64::MAX, f64::MAX, -f64::from_bits(1), f64::from_bits(1)]);
    xs.extend((0..100_000).map(|_| random_f64(&mut rng)));
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for w in xs.windows(2) {
        let (a, b) = (w[0], w[1]);
        if a == b { assert_eq!(key64(a), key64(b), "{a} {b}"); } else { assert!(key64(a) < key64(b), "{a} {b}"); }
    }
}

#[test]
fn threshold_equivalence() {
    let mut rng = Rng(3);
    let mut ts: Vec<f64> = vec![f64::INFINITY, f64::NEG_INFINITY, 1e300, -1e300, 0.0, -0.0, 1e-40, -1e-40];
    for _ in 0..2_000 {
        let x = random_f32(&mut rng) as f64;
        ts.extend([x, x.next_up(), x.next_down(), random_f64(&mut rng)]);
    }
    for &t in &ts {
        let k = key32(f32_at_or_below(t));
        for _ in 0..50 {
            let x = random_f32(&mut rng);
            assert_eq!(x as f64 <= t, key32(x) <= k, "x {x} t {t}");
        }
        let near = f32_at_or_below(t);
        for x in [near, near.next_up(), near.next_down()] { if !x.is_nan() { assert_eq!(x as f64 <= t, key32(x) <= k, "x {x} t {t}"); } }
    }
}

#[test]
fn fill_slots() {
    let data = [f32::NAN, -2.5, 3.0, 7.9, -0.5, 1e10, 1.0, -1.0, 2.0];
    let m = Matrix::from_slice_c(&data, 3, 3);
    let categorical = [false, true, false];
    let mut keys = vec![0i32; 3 * 6];
    fill_keys32(&m, 0, 3, &categorical, &mut keys);
    assert_eq!(&keys[0..6], &[i32::MIN, -1, key32(3.0), i32::MAX, -1, key32(3.0)]);
    assert_eq!(&keys[6..12], &[key32(7.9), 0, key32(1e10), key32(7.9), 0, key32(1e10)]);
    assert_eq!(&keys[12..18], &[key32(1.0), -1, key32(2.0), key32(1.0), -1, key32(2.0)]);
    let cat_big = [0.0f32, 1e10, 0.0];
    let mut k1 = vec![0i32; 6];
    fill_keys32(&Matrix::from_slice_c(&cat_big, 1, 3), 0, 1, &categorical, &mut k1);
    assert_eq!(k1[1], i32::MAX);

    let data64: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let mut keys64 = vec![0i64; 3 * 6];
    fill_keys64(&Matrix::from_slice_c(&data64, 3, 3), 0, 3, &categorical, &mut keys64);
    assert_eq!(&keys64[0..6], &[i64::MIN, -1, key64(3.0), i64::MAX, -1, key64(3.0)]);
    assert_eq!(keys64[7], 0);
}

#[test]
fn fill_strided_matches_contiguous() {
    let mut rng = Rng(4);
    let (rows, cols) = (5, 4);
    let c: Vec<f32> = (0..rows * cols).map(|_| if rng.below(5) == 0 { f32::NAN } else { rng.unit() as f32 - 0.5 }).collect();
    let categorical = [false, true, false, false];
    let mut want = vec![0i32; rows * 2 * cols];
    fill_keys32(&Matrix::from_slice_c(&c, rows, cols), 0, rows, &categorical, &mut want);

    let f: Vec<f32> = (0..rows * cols).map(|i| c[(i % rows) * cols + i / rows]).collect();
    let fm = unsafe { Matrix::from_raw(f.as_ptr(), rows, cols, 1, rows as isize) };
    let mut got = vec![0i32; rows * 2 * cols];
    fill_keys32(&fm, 0, rows, &categorical, &mut got);
    assert_eq!(want, got);

    let wide: Vec<f32> = (0..rows * cols * 2).map(|i| if i % 2 == 0 { c[i / 2] } else { 99.0 }).collect();
    let wm = unsafe { Matrix::from_raw(wide.as_ptr(), rows, cols, (2 * cols) as isize, 2) };
    let mut got = vec![0i32; rows * 2 * cols];
    fill_keys32(&wm, 0, rows, &categorical, &mut got);
    assert_eq!(want, got);

    let rev = unsafe { Matrix::from_raw(c.as_ptr().add((rows - 1) * cols), rows, cols, -(cols as isize), 1) };
    let mut got = vec![0i32; rows * 2 * cols];
    fill_keys32(&rev, 0, rows, &categorical, &mut got);
    for r in 0..rows { assert_eq!(&got[r * 2 * cols..(r + 1) * 2 * cols], &want[(rows - 1 - r) * 2 * cols..(rows - r) * 2 * cols]); }

    let mut part = vec![0i32; 2 * 2 * cols];
    fill_keys32(&fm, 2, 2, &categorical, &mut part);
    assert_eq!(&part[..], &want[2 * 2 * cols..4 * 2 * cols]);
}

#[test]
fn tiny_values_get_the_key_of_zero() {
    let zt = 1e-35f32;
    let data = [1e-36f32, -1e-36, zt, -zt, zt.next_up(), -0.0];
    let mut keys = vec![0i32; 12];
    fill_keys32(&Matrix::from_slice_c(&data, 1, 6), 0, 1, &[false; 6], &mut keys);
    assert_eq!(&keys[..6], &[0, 0, 0, 0, key32(zt.next_up()), 0]);
    let data64: Vec<f64> = vec![1e-36, -1e-36, zt as f64, -(zt as f64), (zt as f64).next_up(), -0.0];
    let mut keys64 = vec![0i64; 12];
    fill_keys64(&Matrix::from_slice_c(&data64, 1, 6), 0, 1, &[false; 6], &mut keys64);
    assert_eq!(&keys64[..6], &[0, 0, 0, 0, key64((zt as f64).next_up()), 0]);
}

#[test]
#[should_panic(expected = "slice too short")]
fn from_slice_c_rejects_overflowing_shapes() {
    let data = [0.0f32];
    let _ = Matrix::from_slice_c(&data, usize::MAX, 2);
}
