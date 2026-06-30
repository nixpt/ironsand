# State

Updated: 2026-06-30T16:30:00-05:00

**Phase-3b nightly-drift migration in progress:** `rustc_codegen_nvvm` targeting nightly-2026-04-02. 7 baseline residuals identified; residual #1 (`IntrinsicResult` migration in `intrinsic.rs`) eliminated via 5-step one-edit-at-a-time discipline. 6 residuals remain: `ThinLtoInput` import, `stable_hash`/`StableHashCtxt`/`HashStable` generics in debug_info/type_map, `scalable_alloca` in builder, and `join_codegen` IndexMap/UnordMap bridge. Phase-3d complete (llvm20/22 plumbing dropped, ~50 LoC removed). Phase-3e complete (obsolete references rationalized). Phase-3c (cfg-gated `verify_module` for i24 bitcast mitigation) pending.

**LLVM 19 pinned as sole codegen backend.** llvm20/22 cargo features removed. CLAUDE.md operative rule: `LLVM_CONFIG_19=/workspace/scratch/llvm19/bin/llvm-config`. ThinLTO stubbed (zero value for NVVM pipeline). DebugInfo scope retained for CUDA-GDB/LLDB-GPU kernel debugging.

**Flash Attention v7 stable on main:** 6.4 TFLOP/s at H=32 (Dh=128, 5070 Ti), +55% over v4 baseline. Phase-8 XOR swizzle on V_T_SMEM eliminates 8-way bank conflicts. GQA and MQA variants also implemented. Br=32 attempted but reverted (warp-specialized softmax needed). `cp.async` pipelining tested & rejected.

**GEMV decode series at plateau** (uncommitted on exp/gemv branch): Q4_K v3 (224-269 GB/s), Q6_K warp (~380 GB/s). Ternary dp4a optimization resolved ALU bottleneck (822 GB/s effective). Kernel families: f32, f16 vec4, int8 dp4a, ternary dp4a, Q4_K warp/fast/v3, Q6_K warp/fast. Next arc likely attention fusion or zorro decode integration.

**Haiku-San crate** (CPU/GPU hybrid orchestrator) extracted as standalone crate. Role-based decode kernels scaffolded (RoleRMSNormSingle, RoleGEMVDecodeSingle, RoleFlashAttnSingle). 4-week implementation plan drafted.


## 2026-06-30T08:26:54-05:00 — annotation
state.md refreshed to reflect current project state: phase-3b drift migration (6/7 residuals remaining), flash attention v7 stable at 6.4 TFLOP/s, LLVM 19 pinned (llvm20/22 plumbing dropped), GEMV plateau, Haiku-San extracted
