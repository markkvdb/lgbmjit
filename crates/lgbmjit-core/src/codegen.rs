//! Cranelift code generation.
//!
//! Trees are grouped by class and cut into blocks. Each block becomes one function that loops over the rows of a tile of key rows (see [`crate::keys`]) and adds the value of each of its trees, in order, to the row's output for that class.
use crate::keys::{f32_at_or_below, key32, key64};
use crate::model::{Error, Forest, MissingType, Tree};
use crate::reference::IterRange;
use crate::settings::Settings;
use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{AbiParam, Block, BlockArg, InstBuilder, MemFlagsData, Type, Value, types};
use cranelift_codegen::settings::{self as cl, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};
use std::collections::HashMap;

/// Input element type. Float32 input uses `i32` keys and float64 input `i64` keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Dtype { F32, F64 }

impl Dtype {
    pub(crate) fn key_type(self) -> Type { match self { Dtype::F32 => types::I32, Dtype::F64 => types::I64 } }

    pub(crate) fn key_bytes(self) -> usize { match self { Dtype::F32 => 4, Dtype::F64 => 8 } }
}

/// A compiled block: `(key rows of a tile, outputs of the tile's first row, rows in the tile)`.
pub(crate) type BlockFn = unsafe extern "C" fn(*const u8, *mut f64, i64);

pub(crate) struct Generated { pub module: JITModule, pub blocks: Vec<BlockFn> }

/// Bitsets of categorical nodes with more than one word, stored where the compiled code reads them.
pub(crate) struct Bitsets { pub words: Vec<u32>, offsets: HashMap<(usize, usize), usize> }

impl Bitsets {
    pub fn collect(forest: &Forest, trees: &[usize]) -> Bitsets {
        let mut words = vec![];
        let mut offsets = HashMap::new();
        for &ti in trees {
            let t = &forest.trees[ti];
            for n in 0..t.num_nodes() {
                if t.is_categorical(n) && t.bitset(n).len() > 1 {
                    offsets.insert((ti, n), words.len());
                    words.extend_from_slice(t.bitset(n));
                }
            }
        }
        Bitsets { words, offsets }
    }
}

struct Ctx<'a> { dtype: Dtype, nf: usize, settings: &'a Settings, bitsets: &'a Bitsets, ptr: Type }

fn compile_err(e: impl std::fmt::Debug) -> Error { Error::Compile(format!("{e:?}")) }

fn new_module() -> Result<JITModule, Error> {
    let mut flags = cl::builder();
    flags.set("opt_level", "speed").map_err(compile_err)?;
    flags.set("enable_verifier", "false").map_err(compile_err)?;
    let isa = cranelift_native::builder().map_err(compile_err)?.finish(cl::Flags::new(flags)).map_err(compile_err)?;
    Ok(JITModule::new(JITBuilder::with_isa(isa, cranelift_module::default_libcall_names())))
}

/// Emits the test of node `n`; the result is true when the row goes left.
fn emit_cond(b: &mut FunctionBuilder, cx: &Ctx, ti: usize, t: &Tree, n: usize, row: Value) -> Value {
    let (kt, ks) = (cx.dtype.key_type(), cx.dtype.key_bytes());
    let f = t.split_feature[n] as usize;
    let trusted = MemFlagsData::trusted();
    if t.is_categorical(n) {
        let words = t.bitset(n);
        let v = b.ins().load(kt, trusted, row, (f * ks) as i32);
        let v = if kt == types::I64 { b.ins().ireduce(types::I32, v) } else { v };
        let in_range = b.ins().icmp_imm_s(IntCC::UnsignedLessThan, v, 32 * words.len() as i64);
        let w = if words.len() == 1 { b.ins().iconst(types::I32, words[0] as i64) } else {
            let idx = b.ins().ushr_imm_u(v, 5);
            let zero = b.ins().iconst(types::I32, 0);
            let idx = b.ins().select(in_range, idx, zero);
            let idx = b.ins().uextend(cx.ptr, idx);
            let off = b.ins().imul_imm_u(idx, 4);
            let base = cx.bitsets.words.as_ptr() as i64 + 4 * cx.bitsets.offsets[&(ti, n)] as i64;
            let base = b.ins().iconst(cx.ptr, base);
            let addr = b.ins().iadd(base, off);
            b.ins().load(types::I32, trusted, addr, 0)
        };
        let sh = b.ins().band_imm_u(v, 31);
        let bit = b.ins().ushr(w, sh);
        let bit = b.ins().band_imm_u(bit, 1);
        let set = b.ins().icmp_imm_s(IntCC::NotEqual, bit, 0);
        return b.ins().band(in_range, set);
    }
    let slot = if t.nan_left(n) { 0 } else { 1 };
    let v = b.ins().load(kt, trusted, row, ((slot * cx.nf + f) * ks) as i32);
    let tk = match cx.dtype { Dtype::F32 => key32(f32_at_or_below(t.threshold[n])) as i64, Dtype::F64 => key64(t.threshold[n]) };
    let le = b.ins().icmp_imm_s(IntCC::SignedLessThanOrEqual, v, tk);
    if t.missing_type(n) != MissingType::Zero { return le; }
    // Inputs within kZeroThreshold of zero have the key 0 (see `keys`), so zero is a key test.
    let is_zero = b.ins().icmp_imm_s(IntCC::Equal, v, 0);
    let dl = b.ins().iconst(types::I8, t.default_left(n) as i64);
    b.ins().select(is_zero, dl, le)
}

/// Emits child `c` without branches: every test in the subtree is computed and selects pick the leaf value.
fn emit_branch_free(b: &mut FunctionBuilder, cx: &Ctx, ti: usize, t: &Tree, c: i32, row: Value) -> Value {
    if c < 0 { return b.ins().f64const(t.leaf_value[!c as usize]); }
    let n = c as usize;
    let cond = emit_cond(b, cx, ti, t, n, row);
    let l = emit_branch_free(b, cx, ti, t, t.left[n], row);
    let r = emit_branch_free(b, cx, ti, t, t.right[n], row);
    b.ins().select(cond, l, r)
}

/// Emits tree `ti` starting in the current block; every path ends by jumping to `join` with the leaf value.
fn emit_tree(b: &mut FunctionBuilder, cx: &Ctx, ti: usize, t: &Tree, row: Value, join: Block) {
    if t.num_nodes() == 0 {
        let v = b.ins().f64const(t.leaf_value[0]);
        b.ins().jump(join, &[BlockArg::Value(v)]);
        return;
    }
    let root_count = t.count(0) as f64;
    let n_leaves = t.n_leaves();
    let start = b.current_block().expect("a current block");
    // Depth first, with the child more training rows reached emitted first, directly after its parent.
    let mut stack: Vec<(i32, Block)> = vec![(0, start)];
    while let Some((c, blk)) = stack.pop() {
        if blk != start { b.switch_to_block(blk); }
        if (t.count(c) as f64) < cx.settings.cold_fraction * root_count { b.set_cold_block(blk); }
        if c < 0 || n_leaves[c as usize] <= cx.settings.branch_free_leaves {
            let v = emit_branch_free(b, cx, ti, t, c, row);
            b.ins().jump(join, &[BlockArg::Value(v)]);
            continue;
        }
        let n = c as usize;
        let cond = emit_cond(b, cx, ti, t, n, row);
        let (l, r) = (t.left[n], t.right[n]);
        let (lb, rb) = (b.create_block(), b.create_block());
        b.ins().brif(cond, lb, &[], rb, &[]);
        if t.count(l) >= t.count(r) {
            stack.push((r, rb));
            stack.push((l, lb));
        } else {
            stack.push((l, lb));
            stack.push((r, rb));
        }
    }
}

/// Builds one function for the trees `trees`, all of class `class`.
fn build_block(
    module: &mut JITModule, fctx: &mut FunctionBuilderContext, cx: &Ctx, forest: &Forest, trees: &[usize], class: usize, name: &str
) -> Result<cranelift_module::FuncId, Error> {
    let ptr = cx.ptr;
    let k = forest.trees_per_iteration;
    let mut ctx = module.make_context();
    for _ in 0..2 { ctx.func.signature.params.push(AbiParam::new(ptr)); }
    ctx.func.signature.params.push(AbiParam::new(types::I64));
    let id = module.declare_function(name, Linkage::Local, &ctx.func.signature).map_err(compile_err)?;
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, fctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        let (head, body, exit) = (b.create_block(), b.create_block(), b.create_block());
        let ri = b.declare_var(types::I64);
        let acc = b.declare_var(types::F64);
        b.switch_to_block(entry);
        let [keys, out, rows] = b.block_params(entry)[..] else { unreachable!() };
        let zero = b.ins().iconst(types::I64, 0);
        b.def_var(ri, zero);
        b.ins().jump(head, &[]);

        b.switch_to_block(head);
        let r = b.use_var(ri);
        let more = b.ins().icmp(IntCC::SignedLessThan, r, rows);
        b.ins().brif(more, body, &[], exit, &[]);

        b.switch_to_block(body);
        let r = b.use_var(ri);
        let roff = b.ins().imul_imm_u(r, (2 * cx.nf * cx.dtype.key_bytes()) as i64);
        let row = b.ins().iadd(keys, roff);
        let ooff = b.ins().imul_imm_u(r, (k * 8) as i64);
        let ooff = b.ins().iadd_imm_u(ooff, (class * 8) as i64);
        let optr = b.ins().iadd(out, ooff);
        let a0 = b.ins().load(types::F64, MemFlagsData::trusted(), optr, 0);
        b.def_var(acc, a0);
        for &ti in trees {
            let join = b.create_block();
            let v = b.append_block_param(join, types::F64);
            emit_tree(&mut b, cx, ti, &forest.trees[ti], row, join);
            b.switch_to_block(join);
            let a = b.use_var(acc);
            let s = b.ins().fadd(a, v);
            b.def_var(acc, s);
        }
        let a = b.use_var(acc);
        b.ins().store(MemFlagsData::trusted(), a, optr, 0);
        let r1 = b.ins().iadd_imm_u(r, 1);
        b.def_var(ri, r1);
        b.ins().jump(head, &[]);

        b.switch_to_block(exit);
        b.ins().return_(&[]);
        b.seal_all_blocks();
        b.finalize(module.target_config());
    }
    module.define_function(id, &mut ctx).map_err(compile_err)?;
    module.clear_context(&mut ctx);
    Ok(id)
}

/// Indices of the trees of class `class` in `range`, in iteration order.
pub(crate) fn class_trees(forest: &Forest, range: IterRange, class: usize) -> Vec<usize> {
    let k = forest.trees_per_iteration;
    (range.start..range.start + range.count).map(|i| i * k + class).collect()
}

/// Compiles the trees in `range` into block functions.
pub(crate) fn generate(forest: &Forest, dtype: Dtype, range: IterRange, settings: &Settings, bitsets: &Bitsets) -> Result<Generated, Error> {
    let mut module = new_module()?;
    let cx = Ctx { dtype, nf: forest.num_feature, settings, bitsets, ptr: module.target_config().pointer_type() };
    let mut fctx = FunctionBuilderContext::new();
    let mut ids = vec![];
    for class in 0..forest.trees_per_iteration {
        let trees = class_trees(forest, range, class);
        for (bi, chunk) in trees.chunks(settings.block_trees.max(1)).enumerate() {
            ids.push(build_block(&mut module, &mut fctx, &cx, forest, chunk, class, &format!("class{class}_block{bi}"))?);
        }
    }
    module.finalize_definitions().map_err(compile_err)?;
    let blocks = ids.iter()
        // SAFETY: each id is a finalized function with the `BlockFn` signature.
        .map(|&id| unsafe { std::mem::transmute::<*const u8, BlockFn>(module.get_finalized_function(id)) })
        .collect();
    Ok(Generated { module, blocks })
}
