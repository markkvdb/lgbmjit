//! Order-preserving integer keys for input values, and the key rows the compiled code reads.
//!
//! A key row holds two slots per feature. For a numerical feature, slot L holds the value's key with NaN as the type's minimum (NaN goes left at every node) and slot R the same key with NaN as the maximum (NaN goes right). Inputs with `|x| <= kZeroThreshold` get the key of 0.0, as LightGBM turns them into 0.0 before the trees. For a categorical feature both slots hold the category: the value truncated toward zero, or -1 for NaN and negative values.
use std::marker::PhantomData;

use crate::reference::ZERO_THRESHOLD;

/// Key of a non-NaN `f32`: `a <= b` exactly when `key32(a) <= key32(b)`. Both zeros map to 0.
pub fn key32(x: f32) -> i32 {
    if x == 0.0 { return 0; }
    let b = x.to_bits() as i32;
    if b < 0 { b ^ 0x7fff_ffff } else { b }
}

/// Key of a non-NaN `f64`: `a <= b` exactly when `key64(a) <= key64(b)`. Both zeros map to 0.
pub fn key64(x: f64) -> i64 {
    if x == 0.0 { return 0; }
    let b = x.to_bits() as i64;
    if b < 0 { b ^ 0x7fff_ffff_ffff_ffff } else { b }
}

/// The largest `f32` that is not above `t`. For every `f32` `x`, `x as f64 <= t` exactly when `x <= f32_at_or_below(t)`.
pub fn f32_at_or_below(t: f64) -> f32 { let f = t as f32; if (f as f64) > t { f.next_down() } else { f } }

/// A read-only 2-D view with element strides, such as a NumPy array.
#[derive(Clone, Copy, Debug)]
pub struct Matrix<'a, T> { ptr: *const T, rows: usize, cols: usize, row_stride: isize, col_stride: isize, _data: PhantomData<&'a T> }

// SAFETY: a Matrix only reads its elements, which outlive it.
unsafe impl<T: Sync> Send for Matrix<'_, T> {}
unsafe impl<T: Sync> Sync for Matrix<'_, T> {}

impl<'a, T: Copy> Matrix<'a, T> {
    /// A view of `rows` x `cols` elements where element (r, c) is at `ptr + r * row_stride + c * col_stride`.
    ///
    /// # Safety
    /// Every such element must be readable and aligned for `T` for the lifetime `'a`.
    pub unsafe fn from_raw(ptr: *const T, rows: usize, cols: usize, row_stride: isize, col_stride: isize) -> Self {
        Matrix { ptr, rows, cols, row_stride, col_stride, _data: PhantomData }
    }

    /// A view of a row-major slice.
    pub fn from_slice_c(data: &'a [T], rows: usize, cols: usize) -> Self {
        assert!(rows.checked_mul(cols).is_some_and(|n| n <= data.len()), "slice too short for {rows} x {cols}");
        unsafe { Self::from_raw(data.as_ptr(), rows, cols, cols as isize, 1) }
    }

    pub fn rows(&self) -> usize { self.rows }

    pub fn cols(&self) -> usize { self.cols }

    /// Element (r, c). Panics when out of range.
    pub fn get(&self, r: usize, c: usize) -> T {
        assert!(r < self.rows && c < self.cols);
        unsafe { *self.ptr.offset(r as isize * self.row_stride + c as isize * self.col_stride) }
    }
}

fn category(x: f64) -> i32 {
    if x.is_nan() { return -1; }
    let v = x as i32;
    if v < 0 { -1 } else { v }
}

macro_rules! fill_keys {
    ($name:ident, $t:ty, $k:ty, $key:ident) => {
        /// Writes the key rows of `rows` input rows starting at `first_row` into `out`, `2 * cols` keys per row.
        pub fn $name(m: &Matrix<$t>, first_row: usize, rows: usize, categorical: &[bool], out: &mut [$k]) {
            let nf = m.cols();
            assert_eq!(categorical.len(), nf);
            assert!(first_row + rows <= m.rows() && out.len() >= rows * 2 * nf);
            for r in 0..rows {
                let row = &mut out[r * 2 * nf..(r + 1) * 2 * nf];
                let (lo, hi) = row.split_at_mut(nf);
                for f in 0..nf {
                    let x = m.get(first_row + r, f);
                    let (l, h) = if categorical[f] {
                        let c = category(x as f64) as $k;
                        (c, c)
                    } else if (x as f64).abs() <= ZERO_THRESHOLD {
                        (0, 0)
                    } else if x.is_nan() {
                        (<$k>::MIN, <$k>::MAX)
                    } else {
                        let k = $key(x);
                        (k, k)
                    };
                    lo[f] = l;
                    hi[f] = h;
                }
            }
        }
    };
}

fill_keys!(fill_keys32, f32, i32, key32);
fill_keys!(fill_keys64, f64, i64, key64);
