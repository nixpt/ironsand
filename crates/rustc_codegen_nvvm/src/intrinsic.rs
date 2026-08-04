use rustc_abi as abi;
use rustc_abi::{self, BackendRepr, Float, HasDataLayout, Primitive, WrappingRange};
use rustc_codegen_ssa::errors::InvalidMonomorphization;
use rustc_codegen_ssa::mir::{IntrinsicResult, operand::OperandValue};
use rustc_codegen_ssa::mir::place::PlaceValue;
use rustc_codegen_ssa::mir::{operand::OperandRef, place::PlaceRef};
use rustc_codegen_ssa::traits::{
    BaseTypeCodegenMethods, BuilderMethods, ConstCodegenMethods, IntrinsicCallBuilderMethods,
    LayoutTypeCodegenMethods, OverflowOp,
};
use rustc_middle::ty::layout::{FnAbiOf, HasTypingEnv, LayoutOf};
use rustc_middle::ty::{self, Ty};
use rustc_middle::{bug, span_bug};
use rustc_span::symbol::kw;
use rustc_span::{Span, Symbol, sym};
use rustc_target::callconv::PassMode;
use tracing::trace;

use crate::abi::LlvmType;
use crate::builder::{Builder, CountZerosKind};
use crate::context::CodegenCx;
use crate::llvm::{self, Type, Value};
use crate::ty::LayoutLlvmExt;

fn handle_128_bit_intrinsic<'ll>(
    b: &mut Builder<'_, 'll, '_>,
    name: Symbol,
    args: &[OperandRef<'_, &'ll Value>],
) -> &'ll Value {
    match name {
        sym::ctlz | sym::cttz => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.ctlz.i128/llvm.cttz.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            let kind = if name == sym::ctlz {
                CountZerosKind::Leading
            } else {
                CountZerosKind::Trailing
            };
            b.emulate_i128_count_zeros(args[0].immediate(), kind, false)
        }
        sym::ctlz_nonzero | sym::cttz_nonzero => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.ctlz.i128/llvm.cttz.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            let kind = if name == sym::ctlz_nonzero {
                CountZerosKind::Leading
            } else {
                CountZerosKind::Trailing
            };
            b.emulate_i128_count_zeros(args[0].immediate(), kind, true)
        }
        sym::ctpop => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.ctpop.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            b.emulate_i128_ctpop(args[0].immediate())
        }
        sym::bswap => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.bswap.i128 (added in LLVM 9.0)
            // When we upgrade NVVM, we can call the real intrinsic directly
            // For now, emulate it by swapping the two i64 halves and byte-swapping each
            b.emulate_i128_bswap(args[0].immediate())
        }
        sym::bitreverse => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.bitreverse.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            b.emulate_i128_bitreverse(args[0].immediate())
        }
        sym::rotate_left | sym::rotate_right => {
            // TODO(@LegNeato): LLVM 7.1 doesn't have llvm.fshl.i128/llvm.fshr.i128
            // When we upgrade NVVM, we can call the real intrinsic directly
            let is_left = name == sym::rotate_left;
            let val = args[0].immediate();
            let shift = args[1].immediate();
            b.emulate_i128_rotate(val, shift, is_left)
        }
        _ => {
            // For any unsupported 128-bit intrinsics, return a fatal error
            // This shouldn't happen with the current set of intrinsics
            b.fatal(format!("unsupported 128-bit intrinsic: {name}"))
        }
    }
}

// llvm 7 does not have saturating intrinsics, so we reimplement them right here.
// This is derived from what rustc used to do before the intrinsics. It should map to the same assembly.
fn saturating_intrinsic_impl<'ll, 'tcx>(
    b: &mut Builder<'_, 'll, 'tcx>,
    width: u32,
    signed: bool,
    is_add: bool,
    args: &[OperandRef<'tcx, &'ll Value>],
) -> &'ll Value {
    use crate::intrinsic::OverflowOp;
    use rustc_codegen_ssa::common::IntPredicate;
    use rustc_middle::ty::IntTy::*;
    use rustc_middle::ty::UintTy::*;

    let tcx = b.tcx;
    let ty = match (signed, width) {
        (true, 8) => Ty::new_int(tcx, I8),
        (true, 16) => Ty::new_int(tcx, I16),
        (true, 32) => Ty::new_int(tcx, I32),
        (true, 64) => Ty::new_int(tcx, I64),
        (true, 128) => Ty::new_int(tcx, I128),
        (false, 8) => Ty::new_uint(tcx, U8),
        (false, 16) => Ty::new_uint(tcx, U16),
        (false, 32) => Ty::new_uint(tcx, U32),
        (false, 64) => Ty::new_uint(tcx, U64),
        (false, 128) => Ty::new_uint(tcx, U128),
        _ => unreachable!(),
    };

    let llty = b.type_ix(width as u64);
    let a = args[0].immediate();
    let c = args[1].immediate();

    // Perform the add or sub, returning the result and an overflow flag
    let (val, ov) = b.checked_binop(
        if is_add {
            OverflowOp::Add
        } else {
            OverflowOp::Sub
        },
        ty,
        a,
        c,
    );

    let zero = b.const_int(llty, 0);

    // Unsigned case: overflow means clamp to either max or min value
    if !signed {
        let all1 = b.not(zero);
        let clamp = if is_add { all1 } else { zero };
        return b.select(ov, clamp, val);
    }

    // Signed case: compute INT_MIN and INT_MAX
    let one = b.const_int(llty, 1);
    let sh = b.const_int(llty, (width - 1) as i64);
    let int_min = b.shl(one, sh);
    let int_max = b.sub(int_min, one);

    // Check if a is negative
    let a_lt0 = b.icmp(IntPredicate::IntSLT, a, zero);

    // Pick the saturation value depending on operation and operand signs
    let sat = if is_add {
        // Add overflow: if a is negative → INT_MIN, else → INT_MAX
        b.select(a_lt0, int_min, int_max)
    } else {
        // Sub overflow: if a is non-negative and c is negative → INT_MAX, else → INT_MIN
        let a_ge0 = b.not(a_lt0);
        let c_lt0 = b.icmp(IntPredicate::IntSLT, c, zero);
        let to_max = b.and(a_ge0, c_lt0);
        b.select(to_max, int_max, int_min)
    };

    // Return the saturation value if overflow, else the computed result
    b.select(ov, sat, val)
}

fn get_simple_intrinsic<'ll>(
    cx: &CodegenCx<'ll, '_>,
    name: Symbol,
) -> Option<(&'ll Type, &'ll Value)> {
    #[rustfmt::skip]
    let llvm_name = match name {
        sym::sqrtf32      => "__nv_sqrtf",
        sym::sqrtf64      => "__nv_sqrt",
        sym::powif32      => "__nv_powif",
        sym::powif64      => "__nv_powi",
        sym::sinf32       => "__nv_sinf",
        sym::sinf64       => "__nv_sin",
        sym::cosf32       => "__nv_cosf",
        sym::cosf64       => "__nv_cos",
        sym::powf32       => "__nv_powf",
        sym::powf64       => "__nv_pow",
        sym::expf32       => "__nv_expf",
        sym::expf64       => "__nv_exp",
        sym::exp2f32      => "__nv_exp2f",
        sym::exp2f64      => "__nv_exp2",
        sym::logf32       => "__nv_logf",
        sym::logf64       => "__nv_log",
        sym::log10f32     => "__nv_log10f",
        sym::log10f64     => "__nv_log10",
        sym::log2f32      => "__nv_log2f",
        sym::log2f64      => "__nv_log2",
        sym::fmaf32       => "__nv_fmaf",
        sym::fmaf64       => "__nv_fma",
        sym::copysignf32  => "__nv_copysignf",
        sym::copysignf64  => "__nv_copysign",
        sym::floorf32     => "__nv_floorf",
        sym::floorf64     => "__nv_floor",
        sym::ceilf32      => "__nv_ceilf",
        sym::ceilf64      => "__nv_ceil",
        sym::truncf32     => "__nv_truncf",
        sym::truncf64     => "__nv_trunc",
        sym::roundf32     => "__nv_roundf",
        sym::roundf64     => "__nv_round",
        sym::round_ties_even_f32 => "__nv_rintf",
        sym::round_ties_even_f64 => "__nv_rint",
        _ => return None,
    };
    trace!("Retrieving nv intrinsic `{:?}`", llvm_name);
    Some(cx.get_intrinsic(llvm_name))
}

fn get_llvm_float_intrinsic<'ll>(
    cx: &CodegenCx<'ll, '_>,
    intrinsic: &str,
    width: u32,
) -> (&'ll Type, &'ll Value) {
    let suffix = match width {
        16 => "f16",
        32 => "f32",
        64 => "f64",
        128 => "f128",
        _ => bug!("unsupported float width {width} for intrinsic {intrinsic}"),
    };
    let name = format!("{intrinsic}.{suffix}");
    cx.get_intrinsic(&name)
}

fn get_llvm_powi_intrinsic<'ll>(cx: &CodegenCx<'ll, '_>, width: u32) -> (&'ll Type, &'ll Value) {
    let suffix = match width {
        16 => "f16",
        32 => "f32",
        64 => "f64",
        128 => "f128",
        _ => bug!("unsupported float width {width} for llvm.powi"),
    };
    let name = format!("llvm.powi.{suffix}.i32");
    cx.get_intrinsic(&name)
}

impl<'ll, 'tcx> IntrinsicCallBuilderMethods<'tcx> for Builder<'_, 'll, 'tcx> {
    fn codegen_intrinsic_call(
        &mut self,
        instance: ty::Instance<'tcx>,
        _fn_abi: &rustc_target::callconv::FnAbi<'tcx, ty::Ty<'tcx>>,
        _args: &[OperandRef<'tcx, &'ll Value>],
        _result_layout: rustc_middle::ty::layout::TyAndLayout<'tcx>,
        _result_place: Option<PlaceValue<&'ll Value>>,
        _span: Span,
    ) -> IntrinsicResult<'tcx, &'ll Value> {
        // TODO: Implement intrinsics for the new signature
        // For now, use fallback mechanism
        IntrinsicResult::Fallback
    }

    fn abort(&mut self) {
        trace!("Generate abort call");
        self.call_intrinsic("llvm.trap", &[]);
    }

    fn assume(&mut self, val: &'ll Value) {
        trace!("Generate assume call with `{:?}`", val);
        self.call_intrinsic("llvm.assume", &[val]);
    }

    fn expect(&mut self, cond: &'ll Value, expected: bool) -> &'ll Value {
        trace!("Generate expect call with `{:?}`, {}", cond, expected);
        self.call_intrinsic("llvm.expect.i1", &[cond, self.const_bool(expected)])
    }

    fn type_checked_load(
        &mut self,
        _llvtable: Self::Value,
        _vtable_byte_offset: u64,
        _typeid: &[u8],
    ) -> Self::Value {
        // LLVM CFI doesnt make sense on the GPU
        self.const_i32(0)
    }

    fn va_start(&mut self, va_list: Self::Value) {
        trace!("Generate va_start `{:?}`", va_list);
        self.call_intrinsic("llvm.va.start", &[va_list]);
    }

    fn retag_mem(&mut self, _val: Self::Value, _info: &rustc_codegen_ssa::common::RetagInfo<Self::Value>) {
        // Not implementing retagging for GPU codegen
    }

    fn retag_reg(&mut self, val: Self::Value, _info: &rustc_codegen_ssa::common::RetagInfo<Self::Value>) -> Self::Value {
        // Not implementing retagging for GPU codegen, just return the value
        val
    }
}
