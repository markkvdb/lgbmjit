//! Running compiled forests over input matrices.
pub use crate::codegen::Dtype;
use crate::codegen::{Bitsets, BlockFn, class_trees, generate};
use crate::keys::{Matrix, fill_keys32, fill_keys64};
use crate::model::{Error, Forest};
use crate::output::finish_rows;
use crate::reference::IterRange;
use crate::settings::Settings;
use cranelift_jit::JITModule;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Input rows, as float32 or float64.
#[derive(Clone, Copy, Debug)]
pub enum Input<'a> { F32(Matrix<'a, f32>), F64(Matrix<'a, f64>) }

impl Input<'_> {
    pub fn rows(&self) -> usize { match self { Input::F32(m) => m.rows(), Input::F64(m) => m.rows() } }

    pub fn cols(&self) -> usize { match self { Input::F32(m) => m.cols(), Input::F64(m) => m.cols() } }

    pub fn dtype(&self) -> Dtype { match self { Input::F32(_) => Dtype::F32, Input::F64(_) => Dtype::F64 } }
}

/// A forest compiled for one input dtype and one iteration range.
pub struct Compiled {
    module: Option<JITModule>,
    blocks: Vec<BlockFn>,
    // Read by the compiled code through baked-in addresses; never modified after compiling.
    _bitsets: Bitsets,
    dtype: Dtype,
    num_feature: usize,
    trees_per_iteration: usize,
    categorical: Vec<bool>,
    settings: Settings,
    tile_rows: AtomicUsize,
    range: IterRange,
}

// SAFETY: the compiled functions only read their arguments and the bitsets, which nothing modifies after compiling. The JIT module is only touched again when it is freed in `drop`.
unsafe impl Send for Compiled {}
unsafe impl Sync for Compiled {}

impl Drop for Compiled {
    fn drop(&mut self) {
        if let Some(m) = self.module.take() {
            // SAFETY: `self` owns the only references to the compiled functions and is being dropped.
            unsafe { m.free_memory() };
        }
    }
}

impl Compiled {
    /// Compiles the trees of `forest` in `range` for input of type `dtype`.
    pub fn compile(forest: &Forest, dtype: Dtype, range: IterRange, settings: &Settings) -> Result<Compiled, Error> {
        let all: Vec<usize> = (0..forest.trees_per_iteration).flat_map(|c| class_trees(forest, range, c)).collect();
        let bitsets = Bitsets::collect(forest, &all);
        let generated = generate(forest, dtype, range, settings, &bitsets)?;
        Ok(Compiled { module: Some(generated.module), blocks: generated.blocks, _bitsets: bitsets, dtype, num_feature: forest.num_feature,
            trees_per_iteration: forest.trees_per_iteration, categorical: forest.categorical.clone(), settings: *settings,
            tile_rows: AtomicUsize::new(settings.tile_rows.max(1)), range })
    }

    pub fn dtype(&self) -> Dtype { self.dtype }

    pub fn range(&self) -> IterRange { self.range }

    /// The settings, with the current tile size.
    pub fn settings(&self) -> Settings { Settings { tile_rows: self.tile_rows(), ..self.settings } }

    /// Rows per tile. It changes speed only, so it can change after compiling.
    pub fn tile_rows(&self) -> usize { self.tile_rows.load(Ordering::Relaxed) }

    pub fn set_tile_rows(&self, rows: usize) { self.tile_rows.store(rows.max(1), Ordering::Relaxed); }

    /// Writes the raw scores of every row of `input` to `out` (`rows * trees_per_iteration` values, row by row), splitting the rows over `threads` threads. Panics when the input does not match the compilation.
    pub fn predict_raw(&self, input: &Input, out: &mut [f64], threads: usize) {
        assert_eq!(input.dtype(), self.dtype, "input dtype does not match the compiled dtype");
        assert_eq!(input.cols(), self.num_feature, "input has the wrong number of columns");
        let (rows, k) = (input.rows(), self.trees_per_iteration);
        assert_eq!(out.len(), rows * k, "output has the wrong length");
        out.fill(0.0);
        if rows == 0 { return; }
        let threads = effective_threads(rows, threads);
        if threads == 1 { self.run(input, 0, out); return; }
        let per = rows.div_ceil(threads);
        std::thread::scope(|s| {
            for (i, chunk) in out.chunks_mut(per * k).enumerate() { s.spawn(move || self.run(input, i * per, chunk)); }
        });
    }

    /// Runs rows `first_row ..` of `input` into `out`, one tile at a time.
    fn run(&self, input: &Input, first_row: usize, out: &mut [f64]) {
        let (nf, k) = (self.num_feature, self.trees_per_iteration);
        let rows = out.len() / k;
        let tile = self.tile_rows().clamp(1, rows);
        match input {
            Input::F32(m) => {
                let mut keys = vec![0i32; tile * 2 * nf];
                for t0 in (0..rows).step_by(tile) {
                    let n = tile.min(rows - t0);
                    fill_keys32(m, first_row + t0, n, &self.categorical, &mut keys);
                    self.run_blocks(keys.as_ptr() as *const u8, &mut out[t0 * k..(t0 + n) * k], n);
                }
            }
            Input::F64(m) => {
                let mut keys = vec![0i64; tile * 2 * nf];
                for t0 in (0..rows).step_by(tile) {
                    let n = tile.min(rows - t0);
                    fill_keys64(m, first_row + t0, n, &self.categorical, &mut keys);
                    self.run_blocks(keys.as_ptr() as *const u8, &mut out[t0 * k..(t0 + n) * k], n);
                }
            }
        }
    }

    fn run_blocks(&self, keys: *const u8, out: &mut [f64], rows: usize) {
        for f in &self.blocks {
            // SAFETY: `keys` holds `rows` key rows of this compilation's layout and `out` holds `rows` output rows.
            unsafe { f(keys, out.as_mut_ptr(), rows as i64) };
        }
    }
}

/// Fewest rows a thread is given: below this, starting a thread costs more than it saves.
pub const MIN_ROWS_PER_THREAD: usize = 256;

/// Threads to use for `rows` rows when `requested` are allowed.
pub fn effective_threads(rows: usize, requested: usize) -> usize { requested.clamp(1, (rows / MIN_ROWS_PER_THREAD).max(1)) }

/// Predictions for every row of `input`: raw scores, then `average_output` and the objective's transform unless `raw_score`.
pub fn predict(forest: &Forest, compiled: &Compiled, input: &Input, raw_score: bool, threads: usize) -> Vec<f64> {
    let mut out = vec![0.0; input.rows() * forest.trees_per_iteration];
    compiled.predict_raw(input, &mut out, threads);
    finish_rows(forest, compiled.range(), raw_score, &mut out);
    out
}
