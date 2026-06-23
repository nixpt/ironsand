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
        let tcx = self.tcx;
        let callee_ty = instance.ty(tcx, self.typing_env());

        let ty::FnDef(def_id, fn_args) = *callee_ty.kind() else {
            bug!("expected fn item type, found {}", callee_ty);
        };

        let sig = callee_ty.fn_sig(tcx);
        let sig = tcx.normalize_erasing_late_bound_regions(self.typing_env(), sig);
        let arg_tys = sig.inputs();
        let ret_ty = sig.output();
        let intrinsic = tcx.intrinsic(def_id).unwrap();
        let name = intrinsic.name;
        let name_str: &str = name.as_str();

        trace!(
            "Beginning intrinsic call: `{:?}`, args: `{:?}`, ret: `{:?}`",
            name, arg_tys, ret_ty
        );

        let llret_ty = self.layout_of(ret_ty).llvm_type(self);

        // Compute fn_abi for intrinsics that need it
        let fn_abi = self.cx.fn_abi_of_instance(instance, ty::List::empty());

        let simple = get_simple_intrinsic(self, name);
        let llval = match name {
            _ if simple.is_some() => {
                let (simple_ty, simple_fn) = simple.unwrap();
                self.call(
                    simple_ty,
                    None,
                    None,
                    simple_fn,
                    &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                    None,
                    Some(instance),
                )
            }
            n if matches!(
                n,
                sym::fabs | sym::minimumf32 | sym::minimumf64 | sym::maximumf32 | sym::maximumf64
            ) || matches!(
                name_str,
                "sqrtf16"
                    | "sqrtf128"
                    | "powif16"
                    | "powif128"
                    | "fmaf16"
                    | "fmaf128"
                    | "copysignf16"
                    | "copysignf128"
                    | "floorf16"
                    | "floorf128"
                    | "ceilf16"
                    | "ceilf128"
                    | "truncf16"
                    | "truncf128"
                    | "roundf16"
                    | "roundf128"
                    | "round_ties_even_f16"
                    | "round_ties_even_f128"
                    | "minimumf16"
                    | "minimumf128"
                    | "maximumf16"
                    | "maximumf128"
                    | "minimum_number_nsz_f16"
                    | "minimum_number_nsz_f32"
                    | "minimum_number_nsz_f64"
                    | "minimum_number_nsz_f128"
                    | "maximum_number_nsz_f16"
                    | "maximum_number_nsz_f32"
                    | "maximum_number_nsz_f64"
                    | "maximum_number_nsz_f128"
            ) =>
            {
                let ty = args[0].layout.ty;
                let width = match ty.kind() {
                    ty::Float(ty::FloatTy::F16) => 16,
                    ty::Float(ty::FloatTy::F32) => 32,
                    ty::Float(ty::FloatTy::F64) => 64,
                    ty::Float(ty::FloatTy::F128) => 128,
                    _ => span_bug!(
                        span,
                        "unsupported float intrinsic {name:?} for argument type {ty:?}"
                    ),
                };
                let (simple_ty, simple_fn) = match name {
                    sym::fabs if width == 32 => self.cx.get_intrinsic("__nv_fabsf"),
                    sym::fabs if width == 64 => self.cx.get_intrinsic("__nv_fabs"),
                    sym::fabs => get_llvm_float_intrinsic(self.cx, "llvm.fabs", width),
                    sym::minimumf32 if width == 32 => self.cx.get_intrinsic("__nv_fminf"),
                    sym::minimumf64 if width == 64 => self.cx.get_intrinsic("__nv_fmin"),
                    sym::maximumf32 if width == 32 => self.cx.get_intrinsic("__nv_fmaxf"),
                    sym::maximumf64 if width == 64 => self.cx.get_intrinsic("__nv_fmax"),
                    _ => match name_str {
                        "sqrtf16" | "sqrtf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.sqrt", width)
                        }
                        "powif16" | "powif128" => get_llvm_powi_intrinsic(self.cx, width),
                        "fmaf16" | "fmaf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.fma", width)
                        }
                        "copysignf16" | "copysignf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.copysign", width)
                        }
                        "floorf16" | "floorf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.floor", width)
                        }
                        "ceilf16" | "ceilf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.ceil", width)
                        }
                        "truncf16" | "truncf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.trunc", width)
                        }
                        "roundf16" | "roundf128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.round", width)
                        }
                        "round_ties_even_f16" | "round_ties_even_f128" => {
                            get_llvm_float_intrinsic(self.cx, "llvm.rint", width)
                        }
                        "minimumf16"
                        | "minimumf128"
                        | "minimum_number_nsz_f16"
                        | "minimum_number_nsz_f32"
                        | "minimum_number_nsz_f64"
                        | "minimum_number_nsz_f128" => {
                            if width == 32 {
                                self.cx.get_intrinsic("__nv_fminf")
                            } else if width == 64 {
                                self.cx.get_intrinsic("__nv_fmin")
                            } else {
                                get_llvm_float_intrinsic(self.cx, "llvm.minnum", width)
                            }
                        }
                        "maximumf16"
                        | "maximumf128"
                        | "maximum_number_nsz_f16"
                        | "maximum_number_nsz_f32"
                        | "maximum_number_nsz_f64"
                        | "maximum_number_nsz_f128" => {
                            if width == 32 {
                                self.cx.get_intrinsic("__nv_fmaxf")
                            } else if width == 64 {
                                self.cx.get_intrinsic("__nv_fmax")
                            } else {
                                get_llvm_float_intrinsic(self.cx, "llvm.maxnum", width)
                            }
                        }
                        _ => span_bug!(
                            span,
                            "unsupported float intrinsic {name:?} for argument type {ty:?}"
                        ),
                    },
                };
                self.call(
                    simple_ty,
                    None,
                    None,
                    simple_fn,
                    &args.iter().map(|arg| arg.immediate()).collect::<Vec<_>>(),
                    None,
                    Some(instance),
                )
            }
            sym::is_val_statically_known => {
                // LLVM 7 does not support this intrinsic, so always assume false.
                self.const_bool(false)
            }
            sym::select_unpredictable => {
                // This should set MD_unpredictable on the select instruction, but
                // nvvm ignores it, so just use a normal select.
                let cond = args[0].immediate();
                assert_eq!(args[1].layout, args[2].layout);
                match (args[1].val, args[2].val) {
                    (OperandValue::Ref(true_val), OperandValue::Ref(false_val)) => {
                        assert!(true_val.llextra.is_none());
                        assert!(false_val.llextra.is_none());
                        assert_eq!(true_val.align, false_val.align);
                        let ptr = self.select(cond, true_val.llval, false_val.llval);
                        let selected =
                            OperandValue::Ref(PlaceValue::new_sized(ptr, true_val.align));
                        selected.store(self, result);
                        return Ok(());
                    }
                    (OperandValue::Immediate(_), OperandValue::Immediate(_))
                    | (OperandValue::Pair(_, _), OperandValue::Pair(_, _)) => {
                        let true_val = args[1].immediate_or_packed_pair(self);
                        let false_val = args[2].immediate_or_packed_pair(self);
                        self.select(cond, true_val, false_val)
                    }
                    (OperandValue::ZeroSized, OperandValue::ZeroSized) => return Ok(()),
                    _ => span_bug!(span, "Incompatible OperandValue for select_unpredictable"),
                }
            }
            _ if name_str == "likely" => self.call_intrinsic(
                "llvm.expect.i1",
                &[args[0].immediate(), self.const_bool(true)],
            ),
            sym::unlikely => self.call_intrinsic(
                "llvm.expect.i1",
                &[args[0].immediate(), self.const_bool(false)],
            ),
            kw::Try => {
                let try_func = args[0].immediate();
                let data = args[1].immediate();

                self.call(self.type_i1(), None, None, try_func, &[data], None, None);
                let ret_align = self.data_layout().i32_align;
                self.store(self.const_i32(0), result.val.llval, ret_align)
            }
            sym::breakpoint => {
                // debugtrap is not supported
                return Ok(());
            }
            sym::va_copy => {
                self.call_intrinsic("llvm.va_copy", &[args[0].immediate(), args[1].immediate()])
            }
            sym::va_arg => {
                match result.layout.backend_repr {
                    abi::BackendRepr::Scalar(scalar) => {
                        match scalar.primitive() {
                            Primitive::Int(..) => {
                                if self.cx().size_of(ret_ty).bytes() < 4 {
                                    // `va_arg` should not be called on a integer type
                                    // less than 4 bytes in length. If it is, promote
                                    // the integer to a `i32` and truncate the result
                                    // back to the smaller type.
                                    let promoted_result = self.va_arg(
                                        args[0].immediate(),
                                        self.cx.layout_of(tcx.types.i32).llvm_type(self.cx),
                                    );
                                    self.trunc(promoted_result, llret_ty)
                                } else {
                                    self.va_arg(
                                        args[0].immediate(),
                                        self.cx.layout_of(ret_ty).llvm_type(self.cx),
                                    )
                                }
                            }
                            Primitive::Float(Float::F16) => {
                                bug!("the va_arg intrinsic does not work with `f16`")
                            }
                            Primitive::Float(Float::F64) | Primitive::Pointer(_) => self.va_arg(
                                args[0].immediate(),
                                self.cx.layout_of(ret_ty).llvm_type(self.cx),
                            ),
                            // `va_arg` should never be used with the return type f32.
                            Primitive::Float(Float::F32) => {
                                bug!("the va_arg intrinsic does not work with `f32`")
                            }
                            Primitive::Float(Float::F128) => {
                                bug!("the va_arg intrinsic does not work with `f128`")
                            }
                        }
                    }
                    _ => bug!("the va_arg intrinsic does not work with non-scalar types"),
                }
            }
            sym::volatile_load | sym::unaligned_volatile_load => {
                let ptr = args[0].immediate();
                let load = self.volatile_load(result.layout.llvm_type(self), ptr);
                let align = if name == sym::unaligned_volatile_load {
                    1
                } else {
                    result.layout.align.abi.bytes() as u32
                };
                unsafe {
                    llvm::LLVMSetAlignment(load, align);
                }
                if !result.layout.is_zst() {
                    self.store_to_place(load, result.val);
                }
                return Ok(());
            }
            sym::volatile_store => {
                let dst = args[0].deref(self.cx());
                args[1].val.volatile_store(self, dst);
                return Ok(());
            }
            sym::unaligned_volatile_store => {
                let dst = args[0].deref(self.cx());
                args[1].val.unaligned_volatile_store(self, dst);
                return Ok(());
            }
            sym::prefetch_read_data
            | sym::prefetch_write_data
            | sym::prefetch_read_instruction
            | sym::prefetch_write_instruction => {
                let (rw, cache_type) = match name {
                    sym::prefetch_read_data => (0, 1),
                    sym::prefetch_write_data => (1, 1),
                    sym::prefetch_read_instruction => (0, 0),
                    sym::prefetch_write_instruction => (1, 0),
                    _ => bug!(),
                };
                self.call_intrinsic(
                    "llvm.prefetch",
                    &[
                        args[0].immediate(),
                        self.const_i32(rw),
                        args[1].immediate(),
                        self.const_i32(cache_type),
                    ],
                )
            }
            sym::carrying_mul_add => {
                let (size, signed) = fn_args.type_at(0).int_size_and_signed(self.tcx);

                let wide_llty = self.type_ix(size.bits() * 2);
                let args = args.as_array().unwrap();
                let [a, b, c, d] = args.map(|a| self.intcast(a.immediate(), wide_llty, signed));

                let wide = if signed {
                    let prod = self.unchecked_smul(a, b);
                    let acc = self.unchecked_sadd(prod, c);
                    self.unchecked_sadd(acc, d)
                } else {
                    let prod = self.unchecked_umul(a, b);
                    let acc = self.unchecked_uadd(prod, c);
                    self.unchecked_uadd(acc, d)
                };

                let narrow_llty = self.type_ix(size.bits());
                let low = self.trunc(wide, narrow_llty);
                let bits_const = self.const_uint(wide_llty, size.bits());
                // No need for ashr when signed; LLVM changes it to lshr anyway.
                let high = self.lshr(wide, bits_const);
                // FIXME: could be `trunc nuw`, even for signed.
                let high = self.trunc(high, narrow_llty);

                let pair_llty = self.type_struct(&[narrow_llty, narrow_llty], false);
                let pair = self.const_poison(pair_llty);
                let pair = self.insert_value(pair, low, 0);
                self.insert_value(pair, high, 1)
            }
            sym::unchecked_shl => self.shl(args[0].immediate(), args[1].immediate()),
            sym::unchecked_shr => {
                let lhs = args[0].immediate();
                let rhs = args[1].immediate();
                match arg_tys[0].kind() {
                    ty::Int(_) => self.ashr(lhs, rhs),
                    _ => self.lshr(lhs, rhs),
                }
            }
            sym::ctlz
            | sym::ctlz_nonzero
            | sym::cttz
            | sym::cttz_nonzero
            | sym::ctpop
            | sym::bswap
            | sym::bitreverse
            | sym::rotate_left
            | sym::rotate_right
            | sym::saturating_add
            | sym::saturating_sub => {
                let ty = arg_tys[0];
                if !ty.is_integral() {
                    tcx.dcx()
                        .emit_err(InvalidMonomorphization::BasicIntegerType { span, name, ty });
                    return Ok(());
                }
                let (size, signed) = ty.int_size_and_signed(self.tcx);
                let width = size.bits();
                if name == sym::saturating_add || name == sym::saturating_sub {
                    saturating_intrinsic_impl(
                        self,
                        width as u32,
                        signed,
                        name == sym::saturating_add,
                        args,
                    )
                } else if width == 128 {
                    handle_128_bit_intrinsic(self, name, args)
                } else {
                    match name {
                        sym::ctlz | sym::cttz => {
                            let y = self.const_bool(false);
                            let llvm_name = format!("llvm.{name}.i{width}");
                            self.call_intrinsic(&llvm_name, &[args[0].immediate(), y])
                        }
                        sym::ctlz_nonzero | sym::cttz_nonzero => {
                            let y = self.const_bool(true);
                            let llvm_name = format!("llvm.{}.i{width}", &name_str[..4]);
                            self.call_intrinsic(&llvm_name, &[args[0].immediate(), y])
                        }
                        sym::ctpop => self.call_intrinsic(
                            &format!("llvm.ctpop.i{width}"),
                            &[args[0].immediate()],
                        ),
                        sym::bswap => {
                            if width == 8 {
                                args[0].immediate() // byte swap a u8/i8 is just a no-op
                            } else {
                                self.call_intrinsic(
                                    &format!("llvm.bswap.i{width}"),
                                    &[args[0].immediate()],
                                )
                            }
                        }
                        sym::bitreverse => self.call_intrinsic(
                            &format!("llvm.bitreverse.i{width}"),
                            &[args[0].immediate()],
                        ),
                        sym::rotate_left | sym::rotate_right => {
                            let is_left = name == sym::rotate_left;
                            let val = args[0].immediate();
                            let raw_shift = args[1].immediate();
                            // rotate = funnel shift with first two args the same
                            let llvm_name =
                                &format!("llvm.fsh{}.i{}", if is_left { 'l' } else { 'r' }, width);

                            // llvm expects shift to be the same type as the values, but rust
                            // always uses `u32`.
                            let raw_shift = self.intcast(raw_shift, self.val_ty(val), false);

                            self.call_intrinsic(llvm_name, &[val, val, raw_shift])
                        }
                        sym::saturating_add | sym::saturating_sub => {
                            let is_add = name == sym::saturating_add;
                            let lhs = args[0].immediate();
                            let rhs = args[1].immediate();
                            let llvm_name = &format!(
                                "llvm.{}{}.sat.i{}",
                                if signed { 's' } else { 'u' },
                                if is_add { "add" } else { "sub" },
                                width
                            );
                            self.call_intrinsic(llvm_name, &[lhs, rhs])
                        }
                        _ => unreachable!(),
                    }
                }
            }
            sym::raw_eq => {
                use rustc_codegen_ssa::common::IntPredicate;
                let tp_ty = fn_args.type_at(0);
                let layout = self.layout_of(tp_ty).layout;
                let use_integer_compare = match layout.backend_repr() {
                    BackendRepr::Scalar(_) | BackendRepr::ScalarPair(_, _) => true,
                    BackendRepr::SimdVector { .. } => false,
                    BackendRepr::SimdScalableVector { .. } => {
                        tcx.dcx()
                            .emit_err(InvalidMonomorphization::NonScalableType {
                                span,
                                name: sym::raw_eq,
                                ty: tp_ty,
                            });
                        return Ok(());
                    }
                    BackendRepr::Memory { .. } => {
                        // For rusty ABIs, small aggregates are actually passed
                        // as `RegKind::Integer` (see `FnAbi::adjust_for_abi`),
                        // so we re-use that same threshold here.
                        layout.size <= self.data_layout().pointer_size() * 2
                    }
                };

                let a = args[0].immediate();
                let b = args[1].immediate();
                if layout.size.bytes() == 0 {
                    self.const_bool(true)
                } else if use_integer_compare {
                    let integer_ty = self.type_ix(layout.size.bits());
                    let ptr_ty = self.type_ptr_to(integer_ty);
                    let a_ptr = self.bitcast(a, ptr_ty);
                    let a_val = self.load(integer_ty, a_ptr, layout.align.abi);
                    let b_ptr = self.bitcast(b, ptr_ty);
                    let b_val = self.load(integer_ty, b_ptr, layout.align.abi);
                    self.icmp(IntPredicate::IntEQ, a_val, b_val)
                } else {
                    let i8p_ty = self.type_i8p();
                    let a_ptr = self.bitcast(a, i8p_ty);
                    let b_ptr = self.bitcast(b, i8p_ty);
                    let n = self.const_usize(layout.size.bytes());
                    let cmp = self.call_intrinsic("memcmp", &[a_ptr, b_ptr, n]);
                    self.icmp(IntPredicate::IntEQ, cmp, self.const_i32(0))
                }
            }
            sym::compare_bytes => self.call_intrinsic(
                "memcmp",
                &[
                    args[0].immediate(),
                    args[1].immediate(),
                    args[2].immediate(),
                ],
            ),

            sym::black_box => {
                args[0].val.store(self, result);
                let result_val_span = [result.val.llval];
                // We need to "use" the argument in some way LLVM can't introspect, and on
                // targets that support it we can typically leverage inline assembly to do
                // this. LLVM's interpretation of inline assembly is that it's, well, a black
                // box. This isn't the greatest implementation since it probably deoptimizes
                // more than we want, but it's so far good enough.
                //
                // For zero-sized types, the location pointed to by the result may be
                // uninitialized. Do not "use" the result in this case; instead just clobber
                // the memory.
                let (constraint, inputs): (&str, &[_]) = if result.layout.is_zst() {
                    ("~{memory}", &[])
                } else {
                    ("r,~{memory}", &result_val_span)
                };
                crate::asm::inline_asm_call(
                    self,
                    "",
                    constraint,
                    inputs,
                    self.type_void(),
                    true,
                    false,
                    llvm::AsmDialect::Att,
                    &[span],
                )
                .unwrap_or_else(|| bug!("failed to generate inline asm call for `black_box`"));

                // We have copied the value to `result` already.
                return Ok(());
            }

            // is this even supported by nvvm? i did not find a definitive answer
            _ if name_str.starts_with("simd_") => todo!("simd intrinsics"),
            // Fall back to a fallback intrinsic implementation, if possible
            _ => {
                // This piece of code was adapted from `rustc_codegen_cranelift`.
                if intrinsic.must_be_overridden {
                    span_bug!(
                        span,
                        "intrinsic {} must be overridden by codegen_nvvm, but isn't",
                        intrinsic.name,
                    );
                }
                return Err(rustc_middle::ty::Instance::new_raw(
                    instance.def_id(),
                    instance.args,
                ));
            }
        };
        trace!("Finish intrinsic call: `{:?}`", llval);
        if !fn_abi.ret.is_ignore() {
            if let PassMode::Cast { cast, .. } = &fn_abi.ret.mode {
                let ptr_llty = self.type_ptr_to(cast.llvm_type(self));
                let ptr = self.pointercast(result.val.llval, ptr_llty);
                self.store(llval, ptr, result.val.align);
            } else {
                OperandRef::from_immediate_or_packed_pair(self, llval, result.layout)
                    .val
                    .store(self, result);
            }
        }
        Ok(())
    }
