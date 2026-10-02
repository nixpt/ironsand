# State

Updated: 2026-06-30

**Phase-3b nightly-drift migration COMPLETE.** All 7 `rustc_codegen_nvvm` residuals targeting nightly-2026-04-02 have been eliminated. `cargo check -p rustc_codegen_nvvm --features llvm19` compiles clean (0 errors, 3 pre-existing warnings: unused `dep_graph` import, unreachable call after `dcx.fatal()`, unused `load_serialized_module_for_thin_lto`). 6/6 unit tests pass (ptx_filter).

Residuals resolved:
- Residual #1 — `IntrinsicResult` migration in `intrinsic.rs`
- Residual #2 — `ThinLtoInput` import removal + `codegen_crate(crate_info)` 3rd arg
- Residual #3 — `StableHash`/`StableHashCtxt`/`HashStable` generics in `type_map.rs` (manual impl for `UniqueTypeId`)
- Residual #4 — `scalable_alloca` signature update in `builder.rs`
- Companion — `CastTarget::llvm_type` flat_map fix in `abi.rs`
- Companion — `valid_range`/`start` field-vs-method fixes in `enums.rs`
- Companion — `c_variadic` field access in `intrinsic.rs`
- Companion — `run_thin_lto` signature alignment in `lib.rs`/`lto.rs`

Phase-3d complete (llvm20/22 plumbing dropped, ~50 LoC removed). Phase-3e complete (obsolete references rationalized). **Phase-3c verify-gate LANDED** (`back.rs`: `verify_module` now `cfg(not(feature = "llvm19"))`; backend check clean, 6/6 tests pass) — but end-to-end exposes the NEXT blocker in the same i24 family: `merge_llvm_modules` → `LLVMRustParseBitcodeForLTO` fails on 6 core CGUs + 1 glam CGU (`Invalid cast (Producer: 'LLVM19.1.7' ...)`; nightly-2026-04-02 core emits a pattern LLVM 19.1.7's reader rejects). Unblocks only together with the `probe/kernel-features` feature-forward fix (also confirmed: its `rustc_codegen_nvvm/llvm19` flag resolves for NO kernel subcrate — `vecadd-kernels` has no backend edge, so the "via cuda_std" premise in the comment is wrong on current HEAD).

**LLVM 19 pinned as sole codegen backend.** llvm20/22 cargo features removed. CLAUDE.md operative rule: `LLVM_CONFIG_19=/workspace/scratch/llvm19/bin/llvm-config`. ThinLTO stubbed (zero value for NVVM pipeline). DebugInfo scope retained for CUDA-GDB/LLDB-GPU kernel debugging.

**Flash Attention v7 stable on main:** 6.4 TFLOP/s at H=32 (Dh=128, 5070 Ti), +55% over v4 baseline. Phase-8 XOR swizzle on V_T_SMEM eliminates 8-way bank conflicts. GQA and MQA variants also implemented. Br=32 attempted but reverted (warp-specialized softmax needed). `cp.async` pipelining tested & rejected.

**GEMV decode series at plateau** (on exp/gemv branch): Q4_K v3 (224-269 GB/s), Q6_K warp (~380 GB/s). Ternary dp4a optimization resolved ALU bottleneck (822 GB/s effective). Kernel families: f32, f16 vec4, int8 dp4a, ternary dp4a, Q4_K warp/fast/v3, Q6_K warp/fast.

**Haiku-San** (CPU/GPU hybrid orchestrator) lives in its own repo, outside ironsand; no ironsand crate depends on it. Role-based decode kernels scaffolded (RoleRMSNormSingle, RoleGEMVDecodeSingle, RoleFlashAttnSingle).


## 2026-06-30 — annotation
state.md updated: phase-3b marked complete. All 7 nightly-drift residuals resolved. rustc_codegen_nvvm compiles clean. Phase-3c (verify_module gate) is the next pending item.
