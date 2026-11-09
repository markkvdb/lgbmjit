//! The `lgbmjit._lgbmjit` extension module: thin bindings over `lgbmjit-core`.
use lgbmjit_core::keys::Matrix;
use lgbmjit_core::model::{Error, Forest as CoreForest};
use lgbmjit_core::reference::IterRange;
use lgbmjit_core::runtime::{Compiled, Dtype, Input, predict};
use lgbmjit_core::settings::Settings;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::Arc;

fn to_py_err(e: Error) -> PyErr {
    match e {
        Error::Parse(_) => PyValueError::new_err(e.to_string()), Error::Unsupported(_) => PyNotImplementedError::new_err(e.to_string()),
        Error::Compile(_) => PyRuntimeError::new_err(e.to_string())
    }
}

/// Rejects arrays the compiled code cannot read element by element: unaligned data, or byte strides that are not whole multiples of the element size (such as a column of a structured array).
fn check_layout<T: numpy::Element>(a: &Bound<'_, PyArray2<T>>) -> PyResult<()> {
    let size = std::mem::size_of::<T>() as isize;
    let aligned = (a.data() as usize).is_multiple_of(std::mem::align_of::<T>());
    if !aligned || a.strides().iter().any(|s| s % size != 0) {
        return Err(PyValueError::new_err(
            "the array must be aligned, with strides that are whole multiples of the item size; \
             pass numpy.ascontiguousarray(data)",
        ));
    }
    Ok(())
}

fn dtype_name(d: Dtype) -> &'static str { match d { Dtype::F32 => "float32", Dtype::F64 => "float64" } }

/// A parsed LightGBM model.
#[pyclass(frozen, module = "lgbmjit._lgbmjit")]
struct Forest { inner: Arc<CoreForest> }

#[pymethods]
impl Forest {
    #[new]
    fn new(model_str: &str) -> PyResult<Self> { Ok(Forest { inner: Arc::new(CoreForest::parse(model_str).map_err(to_py_err)?) }) }

    #[getter]
    fn num_feature(&self) -> usize { self.inner.num_feature }

    #[getter]
    fn num_class(&self) -> usize { self.inner.num_class }

    #[getter]
    fn trees_per_iteration(&self) -> usize { self.inner.trees_per_iteration }

    #[getter]
    fn num_trees(&self) -> usize { self.inner.trees.len() }

    #[getter]
    fn feature_names(&self) -> Vec<String> { self.inner.feature_names.clone() }

    #[getter]
    fn pandas_categorical(&self) -> Option<String> { self.inner.pandas_categorical.clone() }

    #[getter]
    fn text(&self) -> &str { &self.inner.text }

    /// Compiles the iterations LightGBM would use for `start_iteration` and `num_iteration`, for input of `dtype`.
    #[pyo3(signature = (dtype, start_iteration, num_iteration, block_trees, tile_rows, branch_free_leaves, cold_fraction))]
    #[allow(clippy::too_many_arguments)]
    fn compile(
        &self, py: Python<'_>, dtype: &str, start_iteration: i64, num_iteration: i64, block_trees: usize, tile_rows: usize, branch_free_leaves: u32,
        cold_fraction: f64) -> PyResult<CompiledForest> {
        let dtype = match dtype {
            "float32" => Dtype::F32, "float64" => Dtype::F64,
            other => { return Err(PyTypeError::new_err(format!("dtype must be float32 or float64, not {other}"))); }
        };
        if block_trees == 0 || tile_rows == 0 { return Err(PyValueError::new_err("block_trees and tile_rows must be positive")); }
        let range = IterRange::resolve(&self.inner, start_iteration, num_iteration);
        let settings = Settings { block_trees, tile_rows, branch_free_leaves, cold_fraction };
        let forest = self.inner.clone();
        let compiled = py.detach(move || Compiled::compile(&forest, dtype, range, &settings)).map_err(to_py_err)?;
        Ok(CompiledForest { forest: self.inner.clone(), compiled })
    }
}

/// A model compiled for one input dtype and iteration range.
#[pyclass(frozen, module = "lgbmjit._lgbmjit")]
struct CompiledForest { forest: Arc<CoreForest>, compiled: Compiled }

impl CompiledForest {
    fn run<'py>(&self, py: Python<'py>, input: Input, raw_score: bool, threads: usize) -> PyResult<Bound<'py, PyAny>> {
        if input.cols() != self.forest.num_feature {
            return Err(PyValueError::new_err(format!("the data has {} columns, but the model has {} features", input.cols(), self.forest.num_feature)));
        }
        let rows = input.rows();
        let out = py.detach(|| predict(&self.forest, &self.compiled, &input, raw_score, threads));
        let out = PyArray1::from_vec(py, out);
        let k = self.forest.trees_per_iteration;
        Ok(if k == 1 { out.into_any() } else { out.reshape([rows, k])?.into_any() })
    }
}

#[pymethods]
impl CompiledForest {
    /// Predictions for a 2-D float32 or float64 array of any memory layout, matching the compiled dtype.
    fn predict<'py>(&self, py: Python<'py>, data: &Bound<'py, PyAny>, raw_score: bool, num_threads: usize) -> PyResult<Bound<'py, PyAny>> {
        let want = dtype_name(self.compiled.dtype());
        if let Ok(a) = data.cast::<PyArray2<f32>>() {
            check_layout(a)?;
            if self.compiled.dtype() != Dtype::F32 { return Err(PyTypeError::new_err(format!("expected a {want} array, got float32"))); }
            let ro = a.readonly();
            let v = ro.as_array();
            // SAFETY: the view's elements stay alive and unchanged while `ro` borrows the array.
            let m = unsafe { Matrix::from_raw(v.as_ptr(), v.nrows(), v.ncols(), v.strides()[0], v.strides()[1]) };
            return self.run(py, Input::F32(m), raw_score, num_threads);
        }
        if let Ok(a) = data.cast::<PyArray2<f64>>() {
            check_layout(a)?;
            if self.compiled.dtype() != Dtype::F64 { return Err(PyTypeError::new_err(format!("expected a {want} array, got float64"))); }
            let ro = a.readonly();
            let v = ro.as_array();
            // SAFETY: as above.
            let m = unsafe { Matrix::from_raw(v.as_ptr(), v.nrows(), v.ncols(), v.strides()[0], v.strides()[1]) };
            return self.run(py, Input::F64(m), raw_score, num_threads);
        }
        Err(PyTypeError::new_err(format!("expected a 2-D {want} NumPy array")))
    }

    /// Rows per tile; changing it affects speed only.
    #[getter]
    fn tile_rows(&self) -> usize { self.compiled.tile_rows() }

    fn set_tile_rows(&self, rows: usize) -> PyResult<()> {
        if rows == 0 { return Err(PyValueError::new_err("tile_rows must be positive")); }
        self.compiled.set_tile_rows(rows);
        Ok(())
    }

    #[getter]
    fn dtype(&self) -> &'static str { dtype_name(self.compiled.dtype()) }
}

/// The default code generation settings for this machine.
#[pyfunction]
fn default_settings(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let s = Settings::default_for_host();
    let d = PyDict::new(py);
    d.set_item("block_trees", s.block_trees)?;
    d.set_item("tile_rows", s.tile_rows)?;
    d.set_item("branch_free_leaves", s.branch_free_leaves)?;
    d.set_item("cold_fraction", s.cold_fraction)?;
    Ok(d)
}

#[pymodule]
mod _lgbmjit {
    #[pymodule_export]
    use super::{CompiledForest, Forest, default_settings};
    use pyo3::prelude::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add("__version__", env!("CARGO_PKG_VERSION")) }
}
