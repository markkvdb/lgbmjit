//! Compile LightGBM models to machine code with Cranelift and predict with them.
mod codegen;
pub mod keys;
pub mod model;
pub mod output;
pub mod reference;
pub mod runtime;
pub mod settings;
