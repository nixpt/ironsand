phase-3b residual close: all 7 nightly-2026-04-02 drift errors in rustc_codegen_nvvm eliminated

Final sweep completing the Tier-3 drift migration. 7 distinct errors across
6 source files resolved via upstream trait/type alignment:

- lib.rs: pass crate_info as 3rd arg to codegen_crate (upstream now expects
  3 args); update join_codegen return type to FxIndexMap; remove ThinLtoInput
  import and simplify run_thin_lto signature to accept Vec<(String, ModuleBuffer)>
- lto.rs: align run_thin with updated lib.rs signatures
- abi.rs: fix CastTarget llvm_type - use flat_map to handle Option<Reg> prefix
  entries (skipping None)
- builder.rs: update scalable_alloca signature from (layout, ty, align) to
  (elt: u64, align: Align, element_ty: Ty<'_>)
- enums.rs: NicheInfo.valid_range is a field (not method); valid_range.start/end
  are WrappingRange public fields; RangeInclusive::start() is a public method
- intrinsic.rs: FnSig.c_variadic is a field (the self-taking method is only on
  Binder<FnSig>)
- type_map.rs: remove #[derive(StableHash)] on UniqueTypeId (macro removed
  upstream); manual StableHash impl targeting StableHashingContext
