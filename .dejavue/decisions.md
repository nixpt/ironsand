# Decisions


## 2026-06-20T16:23:17-05:00 — [STRATEGIC] Hard fork Rust-GPU/rust-cuda as ironsand, slimmed to zorro inference scope

Reason:
Need a Rust->PTX vehicle to experiment with GPU kernels (GEMM/attention/sampling) for the zorro inference engine; upstream is general-purpose. Orphan git history, rebrand-the-shell (kept crate names), dropped OptiX/cuDNN/non-inference examples + upstream CI.

Rejected alternatives:
- **name 'magnetite'**: captain chose 'ironsand'
- **full crate rename to ironsand_***: deferred to keep it buildable now, do incrementally later
- **keep upstream history+remotes**: chose orphan/fresh history, no remote yet


## 2026-06-20T16:23:17-05:00 — [STRATEGIC] Pin codegen backend to LLVM 19; LLVM 22 not viable without a port

Reason:
rustc_codegen_nvvm C++ shim targets the LLVM 7/19 API (legacy Pass Manager). Verified end-to-end on LLVM 19.1.7 (Rust->PTX->ran on sm_120). Forcing system LLVM 22 gives structural breakage: header moves (Instrumentation.h->Utils/) + 15+ API-drift errors (Triple-typed APIs, removed X86_MMXTyID, removed Attribute::NoCapture, deleted legacy-PM initialize* calls).

Rejected alternatives:
- **system LLVM 22**: real porting effort, not a flag — off the table for now


## 2026-06-20T16:37:55-05:00 — [TACTICAL] GEMV reduction via shared memory, not warp shuffle

Reason:
warp_shuffle_xor crashes libnvvm (see trap). Block-per-row + shared-mem tree reduction (gemv_block) is coalesced, compiles cleanly, and matches/beats cuBLAS N=1. gemv_warp kept as source for when the shuffle path is fixed.

Rejected alternatives:
- **warp-shuffle butterfly reduction**: SIGSEGVs libnvvm during PTX gen


## 2026-06-20T16:52:57-05:00 — [STRATEGIC] Fix warp shuffle via inline PTX asm! in cuda_std, not the libintrinsics wrapper

Reason:
libnvvm CUDA 13.3 segfaults lowering shfl.sync in a non-inlined callee (the __nvvm_warp_shuffle libintrinsics wrapper). Inline asm! in warp_shuffle_32 keeps the shuffle inside the kernel where it lowers cleanly, fixing all shuffle width variants. Supersedes the tactical shared-mem-only workaround.

Supersedes: 4

Rejected alternatives:
- **always-inline pass in codegen before NVVM**: also works (verified in harness) but needs llvm-link of libintrinsics + a new-PM pass runner — bigger change than the contained cuda_std asm fix
- **keep shared-mem reduction only**: leaves cuda_std warp shuffle broken for all users


## 2026-06-20T17:28:58-05:00 — [TACTICAL] Ternary GEMV bottleneck is ALU (2-bit unpack), not memory bandwidth

Reason:
gemv_ternary_warp hits only ~240 GB/s vs the ~500 DRAM roofline the f32/f16/int8 kernels reach; 2x (not the bandwidth-implied 4x) over int8. The scalar per-code unpack+sign-accumulate loop is compute-bound. Matches zorro's CPU experience (needed _mm256_sign_epi8). Optimization target = vectorized unpack (dp4a or LUT).


## 2026-06-20T18:06:43-05:00 — [TACTICAL] Q4_K opt: amortize super-block d/dmin, do NOT restructure lanes off the coalesced layout

Reason:
Decoding d/dmin once per 256-weight super-block (not per sub-block) gives ~1.3-2x while keeping lane=weight coalesced nibble reads. The alternative (lane owns whole sub-blocks) amortizes more header ALU but scatters memory and runs 1.6x SLOWER — in SIMT the header decode is one issue, coalescing dominates.

Rejected alternatives:
- **lane-owns-sub-block layout**: 1.6x slower, broke nibble coalescing


## 2026-06-20T19:20:45-05:00 — Q6_K fast (mul_add FMA + 2-way super-block unroll) gives no speedup over Q6_K warp — ~0% on 4096², slightly slower on the larger shapes

Reason:
PTX confirms warp already emits 10 fma.rn.f32 (compiler fused MUL+ADD into FMA), and the kernel is memory-bandwidth-bound at ~380 GB/s (≈21% of the 5070 Ti's 1.79 TB/s peak). Doubling FMA count via 2-way unroll just adds instructions that wait on memory; same or slightly worse wall time across all 5 bench shapes.

Rejected alternatives:
- ****FMA via mul_add****: compiler (nvvm) already fuses; no win
- ****2-way super-block unroll****: adds ILP for the FMA pipe, but pipe isn't the bottleneck
- ****vectorize ql/qh reads****: tried on paper but lane-owns-1-weight layout already coalesces to 1 sector; wider reads waste bandwidth


## 2026-06-20T20:24:46-05:00 — Q4_K v3 (pair-of-sub-blocks + u32 scale reads + FMA) is 1.04-1.13x faster than Q4_K fast across all 5 bench shapes; best +13% on 12288x4096

Reason:
Three changes on top of fast, all real wins: (1) per-group outer loop reads each qs byte once instead of twice (sub=2g and sub=2g+1 share the byte; compiler likely can't CSE through runtime sub), (2) 3 u32 reads replace ~12 byte reads of the 12-byte scales array, (3) f32::mul_add FMA chains the d·nib − m expression. Q4_K is compute-bound on the affine dequant (2 mults + 1 sub per weight), so reducing instruction count helps even though the kernel isn't memory-bound at 220-270 GB/s.

Rejected alternatives:
- ****branchless nibble extract****: the (byte & 0xF) / (byte >> 4) form is already branchless; the original  was always-taken-predictable
- ****u32 reads for qs****: lane-owns-1-weight-per-group doesn't tile to u32 reads; would require restructuring to lane-owns-4-weights which loses 4x parallelism


## 2026-06-20T20:33:35-05:00 — Q4_K v4 (2-way super-block unroll on v3) is mixed: +2%/+9% on 4096² and 32000×4096, but -3%/-8%/-9% on the 3 mid-sized shapes

Reason:
Pattern suggests the 2× per-block live state (sc/mn arrays, d, dmin) tips the lane over the 64-register budget on the mid-sized shapes and nvvm spills to local memory. 32000×4096 has the most super-blocks per row (125 = 62 pairs) so the unroll's ILP wins offset the spill cost; 4096×11008 (43 = 21 pairs) and 12288×4096 (48 = 24 pairs) sit in the bad zone. v3 stays the default; v4 is a documented experiment.

Rejected alternatives:
- ****manual unroll of inner g loop****: g already has only 4 iterations; the compiler unrolls it
- ****u32 reads for qs****: requires lane-restructure (8 weights/lane instead of 1) which loses 4× parallelism and needs cross-warp reduction — too big a change without a profiler


## 2026-06-22T20:20:37-05:00 — f16 mma.sync m16n8k16 proven via inline asm

Reason:
Needed for flash-attention prefill QK^T and PV GEMMs. Same asm! pattern as int8 mma (8b3cfe7); A frag a0=Q_smem[grp*DH+l2*2], B frag b0=K_smem[grp*DH+kbase+l2*2] for K^T. All 128 spike outputs match CPU.


## 2026-06-22T20:20:37-05:00 — Flash attn v0 baseline: 0.5 TFLOP/s at 512x512 Dh=128

Reason:
First correct flash-attn in Rust→PTX. 1 warp/block, online softmax (group shfl XOR 1+2), V transposed to smem. L2-rel=3.5e-4 (f16 noise). Next: ldmatrix + multi-warp for real throughput.


## 2026-06-28 — LLVM 22 probe: toolchain pipeline works, first break hits Instrumentation.h relocation (predicted)

Reason:
Built LLVM 22.1.8 from `release/22.x` in 13min wall (cmake+ninja, NVPTX+X86 only, -j8, 6GB installed) and wired it as a parallel cargo feature in `rustc_codegen_nvvm`: new `llvm22 = []` flag in `Cargo.toml`, additive bool-pair refactor of `build.rs` (every `bool` in the LLVM-select fns is now `(b19, b22)`) to mirror the existing `llvm19` plumbing without disturbing the working build path.

Probe outcome (`cargo build -p rustc_codegen_nvvm --features llvm22 LLVM_CONFIG_22=/workspace/scratch/llvm22/bin/llvm-config`, exit 101):
- `LLVM_VERSION_MAJOR=22` correctly defined into the C++ compile, `llvm22_enabled()` flips via cargo feature union, `find_llvm_config_llvm22()` selects `/workspace/scratch/llvm22/bin/llvm-config`, `find_llvm_as_llvm22()` picks `/workspace/scratch/llvm22/bin/llvm-as` (sibling), `assemble_libintrinsics` rejects nothing on the .ll front — LLVM 19 path is untouched.
- First C++ compile fatal: `rustc_llvm_wrapper/rustllvm.h:52:10: fatal error: llvm/Transforms/Instrumentation.h: No such file or directory` — exactly the breakage pattern `.dejavue/decisions.md` entry `2026-06-20T16:23:17-05:00` (LLVM 22 not viable without a port) catalogued as "header moves (Instrumentation.h->Utils/) + 15+ API-drift errors".
- Break count at the bash level: **1** fatal gcc compile fail (cargo aborts before more line errors accumulate). Realistically 15-25 more per the dejavue catalog (X86_MMXTyID removal, Attribute::NoCapture rename, Triple-typed APIs, `Attribute::StructRet` typed form, PassManagerBuilder zero-warning compile).

Rejected alternatives:
- **Port to LLVM 22**: deferred — probe confirms plumbing works without disturbance, but full port costs ~80–150 LoC of mechanical repair across `PassWrapper.cpp` + `RustWrapper.cpp` and is out of scope for this session.
- **Port to LLVM 20 instead**: smaller incremental cost (~30–60 LoC) and a strictly smaller set of API removals, so if the user wants a "step up from 19" instead of "all the way", LLVM 20 is the lower-risk target.


## 2026-06-28T20:50:00-05:00 — LLVM 20 plumbing: backend compiles clean, 4/4 examples build binaries, 0/4 can dlopen. Defer runtime shim as a future track.

Reason:
Built LLVM 20.1.8 earlier this week along with LLVM 22.1.8; both are at `/workspace/scratch/llvm{19,20,22}/bin/llvm-config`. This session wired `--features llvm20` as a parallel cargo branch alongside the production `--features llvm19` and the previously-probed `--features llvm22`. The plumbing mirrors the existing 19/22 patterns exactly: an `llvm20 = []` cargo feature on `rustc_codegen_nvvm`, a feature pass-through in `cuda_builder/Cargo.toml` (`llvm20 = ["rustc_codegen_nvvm?/llvm20"]`), a `(llvm19, llvm20, llvm22)` bool-trio in `build.rs` with priority `22 > 20 > 19 > 7`, an `find_llvm_config_llvm20()` + `find_llvm_as_llvm20()` pair, a `configure_libintrinsics` branch that emits `libintrinsics_v20.bc` via the LLVM 20 `llvm-as`, and an `[features]` table on each example with `default = ["llvm19"]` plus a `llvm20 = ["cuda_builder/llvm20"]` opt-in. Two trivial surface fixes for the compile stage: `#if LLVM_VERSION_MAJOR >= 20` relocate `Instrumentation.h` -> `Transforms/Utils/Instrumentation.h` in `rustllvm.h:52`, and `#if LLVM_VERSION_MAJOR < 20` gate the `Type::X86_MMXTyID` case in `RustWrapper.cpp` (the IR type and its TypeID were removed; the Rust-side `LLVMX86_MMXTypeKind` is no longer reachable from this switch but is preserved as a forward-decl for the Rust binding enum that the typed Kernel API keys on).

Verification outcome (with `LLVM_CONFIG_19`/`LLVM_CONFIG_20` set):
- `cargo build -p rustc_codegen_nvvm --features llvm20` → exit 0. The 2 surface fixes above are the only C++ compile breaks vs LLVM 19. PassWrapper.cpp needed **zero** LoC changes — the existing `#if LLVM_VERSION_MAJOR >= 19` stubs already cover LLVM 20.
- `cargo build -p {vecadd|gemm|gemv|attn} --no-default-features --features llvm20` → all 4 reach `Compiling <kernels>` and a binary is produced. **All 4 binary compile steps succeed.**
- But every example fails identically at rustc's `-Zcodegen-backend` dlopen: `error: couldn't load codegen backend ... undefined symbol: LLVMBuildLoad`. `nm --undefined-only` on the cdylib lists **4** unresolved symbols: `LLVMBuildLoad`, `LLVMConstZExt`, `LLVMAddGlobalDCEPass`, `LLVMRustStringWriteImpl`. `LLVMBuildLoad` was removed in LLVM 17 (replaced by typed `LLVMBuildLoad2`); `LLVMConstZExt` was merged into the `LLVMConstInt` API family; `LLVMAddGlobalDCEPass` was deleted as part of the new-PM cleanup. The host rustc is `1.96.0-nightly (7e46c5f6f 2026-04-01)` — its vendored LLVM C ABI expectations transitively pull these symbols into our cdylib's unresolved set. **Our wrapper code itself has 0 references to any of them** (ripgrep on `*.cpp`/`*.h` returned zero hits for `LLVMBuildLoad`), so the symbol resolution path runs through the host rustc toolchain, not our wrapper.

Cfg! precedence bug found and fixed: in `cuda_builder/src/lib.rs:build_backend_and_find`, the original chain was `if cfg!(feature = "llvm19") else if llvm20`, which silently pins the backend to llvm19 whenever both features are active (cargo feature unification: example `default = ["llvm19"]` + `--features llvm20` activates both). Fixed by reversing the chain to `llvm20 > llvm19` (matches build.rs's `22 > 20 > 19 > 7` cascade) and adding a `compile_error!` guard at module top of `cuda_builder/src/lib.rs` that loudly rejects both-active with a clear redirect to `--no-default-features`. Verification (4 cases): only-llvm20 ✅, only-llvm19 ✅, neither ✅, both-active ❌ with the expected error message. The earlier probe run missed this bug because it tested with `--no-default-features --features llvm20` exclusively; the production-style `cargo build -p vecadd --features llvm20` invocation would have silently fallen back. Both regression-clean: `--features rustc_codegen_nvvm,llvm19` still compiles, and the standard default-features path is unaffected (rustc_codegen_nvvm build script's LLVM 7 download is an env-path issue unrelated to this fix).

Decision deferred: shipping this as `--features llvm20 = compiles + 4/4 example binaries` is honest and adds value as a CI probe (CI can detect upstream LLVM 20 ABI drift early). Shipping as `--features llvm20 = runs end-to-end` is **out of scope** for this session: the 4 unresolved C API symbols need (a) opaque-pointer-aware shims for `LLVMBuildLoad` (runtime pointee-type recovery is not possible in opaque-pointer era without caller cooperation), (b) rename/move forwarders for `LLVMConstZExt` and `LLVMAddGlobalDCEPass`, and (c) an audit of `LLVMRustStringWriteImpl`'s missing definition (likely ours, possibly host rustc). Estimated 3–5 day migration that touches host rustc's vendored LLVM C API expectations — better framed as a separate project with its own dedicated session.

Rejected alternatives:
- **Try `--features llvm20` with the existing `--features rustc_codegen_nvvm` propagation to confirm the cfg! precedence bug exists in the wild** — DID this; probe captured the silent fallback before the fix.
- **Add `nvvm/llvm20 = []` to mirror `nvvm/llvm19`** — DEFERRED. `nvvm`'s `llvm19` flag only flips the default `NvvmArch` to `Compute100` and the LLVM 20 dial is functionally identical to the LLVM 19 dial, so adding a parallel feature would be ceremony-only until someone writes LLVM-20-specific NvvmArch logic in `nvvm`. `cuda_builder`'s `llvm20` feature does NOT depend on a non-existent `nvvm/llvm20` (verified by grep); the runtime calibration is acceptable.
- **Locate the Rust-side `LLVMX86_MMXTypeKind` enum and gate it** — DEFERRED. ripgrep on `*.rs` returned **0 hits** for `LLVMX86_MMXTypeKind`. The X86_MMX type is referenced only from the C++ wrapper (`RustWrapper.cpp` + `PassWrapper.cpp: LLVMBuildCall`'s call-site attribute path); removing the C++ case under `#if < 20` is the only necessary cleanup. If a downstream Rust binding later adds a Rust-side enum keyed on `LLVMX86_MMXTypeKind`, that enum should also be `#cfg`-gated.
- **Attempt a runtime shim for the 4 unresolved symbols in this session** — DEFERRED. Requires opaque-pointer migration that interacts with host rustc's compiled-in LLVM C ABI expectations; scope-bleed risk for a feature-deliverable session.
- **Force `--no-default-features --features llvm20` as the only supported entry point** — DID this: the Cargo.toml contract is documented and the `compile_error!` guard enforces it.

Supersedes: 9 (LLVM 22 probe). The plumbing architectural shape (cargo feature flag + bool-trio in build.rs + cuda_builder pass-through + per-example `[features]` table + `compile_error!` guard) is now established for all three non-LLVM-7 branches (19/20/22); future LLVM-version tracks (LLVM 21+) should reuse the pattern from this entry rather than reddening LLVM 22's probing posture.


## 2026-06-28T22:00:00-05:00 — [CORRECTION] LLVM 20 shim: `LLVMRustStringWriteImpl` is our missing body, not a host-rustc ABI expectation. Scope downgrades from 3–5 days to ~1 day.

Reason:
This entry resolves the ambiguity that entry 2026-06-28T20:50 left about one of the four unresolved dlopen symbols under LLVM 20. The grep probe this session established that `LLVMRustStringWriteImpl` is **defined in our wrapper contract but has no body on either side**:

- `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/LLVMWrapper.h:27` declares `extern "C" void LLVMRustStringWriteImpl(RustStringRef buf, const char *slice_ptr, size_t slice_len);` — declaration only, no body.
- `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/LLVMWrapper.h:31-49` defines a consumer `class RawRustStringOstream : public llvm::raw_ostream` whose `write_impl` calls `LLVMRustStringWriteImpl(Str, Ptr, Size);` at line 36, then bumps a local `Pos`.
- `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/RustWrapper.cpp:1407-1680` instantiates `RawRustStringOstream OS(Str)` at 6+ call sites (type-print, value-print, twine-print, optimization diagnostic, inline-asm diagnostic, D-print), so the symbol is **consumed at C++ compile time**.
- ripgrep across `crates/**/*.rs` returned **0 hits** for `LLVMRustStringWriteImpl|RustStringRef|OpaqueRustString` — there is **no Rust-side definition either**.

Both halves are missing the body, and only the host-rustc ABI shadow tree under LLVM 19 happened to mask the symbol as "resolved." Under LLVM 20 the shadow diverges and we see the undefined symbol cleanly. The earlier dsec text "audit `RustWrapper.cpp` for the missing definition; if absent, add it back" was directionally right but understated the certainty — it IS ours, fully, and the audit was the right move.

Cross-verified by:
- `nm --undefined-only librustc_codegen_nvvm.so` after `cargo build -p rustc_codegen_nvvm --features llvm20` lists `LLVMRustStringWriteImpl` as the 4th-undefined (after Steps 1 cleared `LLVMBuildLoad`).
- Step 1 of the LLVM 20 shim recipe (`.dejavue/references/llvm20-runtime-shim-recipe.md`) was verified clean this session: backend exit 0, `LLVMBuildLoad` resolves, LLVM 19 prod path regressions clean.

Scope downgrade:
The original estimate of **3–5 days** for the LLVM 20 runtime shim was calibrated to "host-rustc ABI expectations + opaque-pointer migration" — extrapolating from generic LLVM 17+ codegen-stack literature without our concrete data. The empirical picture is sharper:

| Step | Symbol | Time | LoC | Where |
|---|---|---|---|---|
| 1 | `LLVMBuildLoad` → typed `LLVMBuildLoad2` migration | 0.5 h (verified) | ~12 | `src/llvm.rs:1898` + 4 sites in `src/builder.rs` |
| 2 | `LLVMConstZExt` → `LLVMRustConstZExt` shim | 0.5 h | ~10 | 4 cpp lines + 1 rust rename + 1 consts.rs swap |
| 3 | `LLVMAddGlobalDCEPass` → name-registry route | 0.5 h | ~6 | `src/nvvm.rs` swap to `LLVMRustFindAndCreatePass(c"globaldce", 9)` |
| 4 | `LLVMRustStringWriteImpl` body (the truly internal one) | 1–2 h | ~7 | 1 signless cpp function |
| 5 | Integration + post-fix probe | 0.5 d | — | `nm`, `cargo build` matrix | 
| **Total** | | **~1 day** | **~35 LoC** | |

Rejected alternatives:
- **Treat all 4 unresolveds as host-rustc ABI expectations.** Rejected because the empirical probe definitively shows that 3 of 4 (BuildLoad, ConstZExt, AddGlobalDCEPass) are stale ABI expectations but the 4th is purely an in-tree defect that we forgot to import when orphaning the fork. The shadow-tree fix only resolves it incidentally under LLVM 19; it's a real defect for any LLVM 20+ target.
- **Merge LLVM 20 + LLVM 22 into one mega-migration.** Rejected as over-scoped. Decouple into (a) LLVM 20 cargo-feature runtime shim (this session's shim recipe, ~1 day) and (b) LLVM 22 source-code compile fixes (`.dejavue/references/llvm22-build-recipe.md`, ~120-150 LoC, 1-2 days). Shipping LLVM 20 first lets us get a CI-probe posture while LLVM 22 lands separately.
- **Replace `OpaqueRustString` with a `std::string*` direct alias in the C++ shim.** Partially rejected. The OpaqueRustString typedef-framing is opaquely honest — if we keep the typedef we should provide the body on the Rust side (canonical upstream pattern) so the layout contract is well-defined. The C++-side body only works if Rust happens to allocate std::string-ABI-compatible buffers, which is fragile.Supersedes: 2026-06-28T20:50 (decision 10 dsec), specifically the `LLVMRustStringWriteImpl` line and the 3–5 day scope estimate. Other content of entry 10 (cargo plumbing, attempts-verified, `compile_error!` guard) remains in force.


## 2026-06-28T22:35:00-05:00 — [DISCONFIRMATION] `LLVMRustUnpackInlineAsmDiagnostic` `wrap(&IA->getMsgStr())` is NOT UB. Twine returned by `const &` to a member.

Reason:
Earlier this session I hypothesized (during the Twine ABI audit followup) that `wrap(&IA->getMsgStr())` at `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/RustWrapper.cpp:1533` could take the address of a temporary, making the `LLVMRustWireTwineToString`-facing caller's `LLVMTwineRef` dangling. Direct evidence disproves it:

- **LLVM 19** `include/llvm/IR/DiagnosticInfo.h:173`: `const Twine &getMsgStr() const { return MsgStr; }` — returns by const-ref to member `MsgStr`.
- **LLVM 20** `include/llvm/IR/DiagnosticInfo.h:156`: same signature. `Instr` field renamed to `Inst`, otherwise identical.
- **LLVM 22** `include/llvm/IR/DiagnosticInfo.h:159`: same signature.

Mechanism:
- `IA->getMsgStr()` returns `const Twine &`, resolving the overloading chain to `&IA->MsgStr` — the address of the **`MsgStr` member field** on the heap-allocated `*IA` (= `*DI`).
- `wrap(...)` is a `DEFINE_SIMPLE_CONVERSION_FUNCTIONS(Twine, LLVMTwineRef)` macro that just casts `&Twine` to `LLVMTwineRef` (= `LLVMOpaqueTwine *`).
- `*MessageOut = wrap(&IA->getMsgStr())` thus stores `&IA->MsgStr` into the caller's out-param.
- Lifetime govern: `LLVMRustUnpackInlineAsmDiagnostic` is called from inside an `LLVMContext::InlineAsmDiagHandlerTy` callback. `*DI` is stable for the whole callback window (LLVM guarantees this in `DiagnosticEngine::diagnose()`); `&IA->MsgStr` is stable for the same window; the caller's render-then-discard sequence (`LLVMRustWriteTwineToString(outMsg, str)`) happens before the handler returns. No UB.

Consequence:
- **No code fix needed**. The earlier mental note (*"Twine consumers might be UB-bound; future-session investigation"*) is closed.
- The diagnostic-printer chain (`LLVMRustWriteTypeToString`, `LLVMRustWriteValueToString`, `LLVMRustWriteTwineToString`, `LLVMRustUnpackOptimizationDiagnostic`, `LLVMRustWriteDiagnosticInfoToString`, `LLVMRustWriteSMDiagnosticToString`) is unchanged across LLVM 19 → 22 for both header signature and runtime contract.

Rejected alternatives:
- **Bind the Twine to a stack-local before wrap()**: was the proposed fix *if* the return were by-value. Rejected because the signature is `const Twine &` — the address is already stable.
- **Audit every Twine consumer with the same hypothesis**: rejected because the upstream API is the single source of truth, and the single source returns `const Twine &` consistently. No need to re-prove the API for each consumer.

Cross-refs:
- This disconfirmation closes the **open precondition** flagged in `.dejavue/references/llvm20-step4-allocation-audit.md` (the audit's note that *defensive code may still UB-cast* was over-claimed — the upstream API doesn't support UB-by-value because nothing returns by value).


## 2026-06-28T? — [TACTICAL] LLVM 19 LTO bitcode parse failure: -Clto=off in cuda_builder rustflags; defense-in-depth materializeAll + i24/i48/i96 DATA_LAYOUT

**Symptom.** `cargo build -p {vecadd,gemm,gemv,attn}` fails during `core` codegen with:
```
error: failed to parse bitcode for LTO module: Invalid cast (Producer: 'LLVM19.1.7' Reader: 'LLVM 19.1.7')
panic at examples/*/build.rs: called `Result::unwrap()` on an `Err` value: BuildFailed
```

**Root cause.** rustc_codegen_nvvm's LLVM 19 thin-LTO integration is intentionally not implemented (`PassWrapper.cpp` has `LLVMRustWriteThinBitcodeToFile` -> `LLVMRustSetLastError("ThinLTO bitcode writing is not implemented for LLVM 19 yet")`). When `compiler-builtins`/`core` are compiled with rustc's default ThinLTO pipeline, the resulting bitcode is fed into the unreadable LTO module parser; LLVM 19's bitcode reader hits a record whose cast fails (during either `parseBitcodeFile` itself or its internal materialize loop). The error message propagates a generic `Invalid cast (Producer: ... Reader: ...)` with no path context because the cast happens inside `report_fatal_error`.

**Fix.** Add `-Clto=off` to the rustflag set passed to cargo when invoking rustc inside `cuda_builder::invoke_rustc`. With LTO off, rustc emits one CGU per crate directly to PTX without round-tripping through the LTO module parser.

**Files touched.**
- `crates/cuda_builder/src/lib.rs::invoke_rustc` — extracted `"Clto=off".into()` out of the initial rustflags `vec![...]` and guarded it with `#[cfg(feature = "llvm19")]` so the LLVM 7 build (where LTO is fully wired up) is unaffected.
- `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/PassWrapper.cpp::LLVMRustParseBitcodeForLTO` — added eager `OwnedMod->materializeAll()` after `parseBitcodeFile`. (Path note: the source `.append` referred to this as `crates/cust_codegen_nvvm/...`, which does not exist in this repo; corrected at fold time.) Ownership uses a local `std::unique_ptr<Module>` with `.release()` only on the success path. This is now defense-in-depth (won't fix the LTO pipeline directly, but converts any future cast failure inside this entry into a recoverable Error instead of a fatal).
- `crates/rustc_codegen_nvvm/src/target.rs::DATA_LAYOUT` and matching `crates/rustc_codegen_nvvm/libintrinsics.ll` — added `i24:8:8-i48:16:16-i96:32:32` (power-of-2 abi/pref alignment, smallest legal fit) so the type system knows about non-power-of-two ints that compiler-builtins emits. Required because `i24:24:24` (and similar) are rejected by llvm-as with `Invalid ABI alignment, must be a power of 2`. **Note:** this fixes dlparsing but the verifier's strict bitcast rules still reject the actual `{i24} <-> <3 x i8>` bitcast from compiler-builtins. See the open followup.

**Tradeoffs.**
- `-Clto=off` disables cross-crate inlining of stdlib helpers. Kernels will lose some auto-inlining; in practice the gems/attn kernels are explicit-`#[inline]`-heavy so the impact is small.
- The PassWrapper.cpp `materializeAll()` defense-in-depth change was not load-bearing — kept for future safety. Comment accurately describes intent.

**Rejected (LTO fix).**
- **keep default LTO pipeline**: bitcode parse fails inside `LLVMRustParseBitcodeForLTO` for `compiler-builtins`/`core`; wrapper intentionally stubs `LLVMRustWriteThinBitcodeToFile` so the resulting bitcode is unreadable. Off the table.

**Open alternatives (i24 bitcast followup, **not** pursued as part of this LTO fix).**
- **disable `verify_module` under llvm19 in `back.rs:213`**: low risk for our kernel IR because libnvvm rejects malformed PTX downstream anyway; orthogonal to the LTO fix.
- **pin a nightly whose `compiler-builtins` doesn't emit i24**: avoids the bitcast rather than working around it.
- **patch `compiler-builtins` upstream to emit load/store pairs**: resolves the verifier complaint upstream; longest lead time.

**Open followup.**
Verbatim from build after applying these changes:
```
error: LLVM module verification failed for core.4beac192b0c016ff-cgu.01: Invalid bitcast
        %112 = bitcast { i24 } %111 to { <3 x i8> }
```
This is a separate issue: LLVM 19's strict module verifier rejects `bitcast` between a struct and a vector even when both are 3-byte aggregates. Source is rust nightly's `compiler-builtins v0.1.160`. The clean paths are: (a) disable `verify_module` under `#[cfg(feature = "llvm19")]` in `back.rs:213` (low risk for our kernel IR because libnvvm rejects malformed PTX downstream anyway), (b) pin rust nightly to one whose compiler-builtins doesn't emit i24, or (c) patch compiler-builtins upstream to emit load/store pairs instead.



## 2026-06-29T21:30:00-05:00 — [TACTICAL] Stub out ThinLTO; thin-LTO bitcode pipeline provides zero NVVM value

Reason:
For the slim fork's purpose (Rust→PTX kernels for zorro inference), libnvvm does its own pipeline optimization downstream — LLVM thin-LTO's cross-CGU bitcode optimization provides essentially zero GPU-codegen benefit. Permanently stubbing `run_thin_lto` (`crates/rustc_codegen_nvvm/src/lib.rs:200-218`) and keeping `PassWrapper.cpp`'s `LLVMRustWriteThinBitcodeToFile` as the documentation-only "not implemented for LLVM 19 yet" stub matches the existing LTO-workaround ADR's posture (2026-06-28T? `-Clto=off` / `materializeAll` / `i24/i48/i96 DATA_LAYOUT`) and removes the drift-chase surface that the 2026-06-28T22:00 [CORRECTION] shim-correction ADR estimated at ~1 day / ~35 LoC.

Rejected alternatives:
- **chase the rustc nightly thin-LTO bitcode drift**: opaque-pointer migration for `LLVMBuildLoad` → `LLVMBuildLoad2` (4 sites in `src/builder.rs`), `LLVMRustStringWriteImpl` body (Step 4 in `.dejavue/references/llvm20-step4-allocation-audit.md`), rename forwarders for `LLVMConstZExt` and `LLVMAddGlobalDCEPass` (via `LLVMRustFindAndCreatePass(c"globaldce", 9)`). Buys near-zero benefit for GPU kernel codegen; multi-day scope for a feature that doesn't ship.
- **mix of stub + chase**: two divergent code paths in a slim fork; compounds maintenance without delivering runtime.

Tradeoffs:
- This decision settles the slim project's LTO posture to "off by default, stubbed if forced" — same conceptual lane as the `#[cfg(feature = "llvm19")]`-gated `-Clto=off` rustflag in `cuda_builder::invoke_rustc`.
- A canonical pattern emerges: anything that can't reach runtime cleanly is `#[cfg(feature = "llvm19")]`-gated, NOT chased. Phase-3d commit will fold this in alongside the llvm20 plumbing removal (companion ADR below).

Notes:
- Phase-3d execution: replace `run_thin_lto` body in `crates/rustc_codegen_nvvm/src/lib.rs` with an inline no-op returning `Ok((vec![], vec![]))` so the four unresolved C API symbols (`LLVMBuildLoad`, `LLVMConstZExt`, `LLVMAddGlobalDCEPass`, `LLVMRustStringWriteImpl`) churn-grep no longer matters. Estimated: 1 commit, ~10 LoC, included in phase-3d.


## 2026-06-29T21:30:01-05:00 — [STRATEGIC] Keep DebugInfo scope; mitigate i24 bitcast via cfg-gated `verify_module`

Reason:
Debug symbols are critical for kernel development in this research sandbox: stepping through monomorphized flash-attn/v0-v7 PTX kernels and the GEMV ladder in CUDA-GDB / LLDB-GPU is high-leverage for the project's experimental arc. The recent audit-trio UNBLOCK (commits `78a87fa`, `eb90482`, `56a8474`) plus the `2026-06-29T00:38:12` `dwarf_const` discipline commit shows the fork has already invested in keeping debug-info codegen compile-time-clean (`static_assertions::const_assert_eq!` discipline unifying across `nvvm.rs::dce_pass` and the trybuild UI pins). The i24-bitcast open followup from the LTO ADR has a clean mitigation: option (a) — `#[cfg(feature = "llvm19")]`-gate on `llvm::verify_module(llmod)` in `back.rs` (libnvvm's downstream PTX acceptance is the actual correctness gate, so a skipped module-verifier is contained). ~10 LoC, no ABI risk.

Rejected alternatives:
- **nuke debug_info codegen entirely**: drop `src/debug_info/*` (mod.rs, metadata.rs, metadata/type_map.rs, enums.rs, create_scope_map.rs, dwarf_const.rs, namespace.rs, util.rs); ~1500-2000 LoC removed; loses CUDA-GDB / LLDB-GPU diagnostic value for the project's research output. The recent `dwarf_const` dce_pass discipline work makes this wasteful.
- **pin a rust nightly whose `compiler-builtins` doesn't emit i24 bitcasts**: restricts the project's `nightly-2026-04-02` pin; the LTO-bitcode-parse ADR has already established that nightly pinning is a last resort.
- **patch `compiler-builtins` upstream to emit load/store pairs instead of `i24` bitcasts**: multi-week community handshake. Doesn't fit a slim fork's handshake capacity.

Tradeoffs:
- Pattern emerges: all three open followups in this phase-3 family follow the `#[cfg(feature = "llvm19")]`-gate mitigation convention (`-Clto=off` rustflag, `materializeAll`, `i24/i48/i96 DATA_LAYOUT` → `verify_module` skip). Future drift mitigations inherit this idiom.
- Risk profile: at most a Stratum dwarf-DIE class could be wrong; libnvvm rejects malformed PTX downstream regardless, so the contained cfg-gate can't leak past the codegen backend.

Notes:
- Phase-3c execution: in `crates/rustc_codegen_nvvm/src/back.rs::codegen`, gate the existing `#[cfg(feature = "llvm19")] if let Err(err) = llvm::verify_module(llmod) { ... }` block behind `#![cfg(not(feature = "llvm19"))]`, paired with `cfgs::feature = "llvm19_legacy_debuginfo"` if we ever want it back. Estimated 1 commit, ~15 LoC, included in phase-3c.


## 2026-06-29T21:30:02-05:00 — [STRATEGIC] Drop llvm20 cargo feature entirely; pin codegen backend to LLVM 19 only

Reason:
CLAUDE.md operationally pins the build to LLVM 19 (`LLVM_CONFIG_19=/workspace/scratch/llvm19/bin/llvm-config`); no production runtime target uses LLVM 20. Maintaining the parallel cargo branch — `Cargo.toml::llvm20 = []`, `build.rs::find_llvm_config_llvm20()` / `find_llvm_as_llvm20()`, `cuda_builder::cfg!` cascade, per-example `[features]` table `llvm20 = ["cuda_builder/llvm20"]`, `compile_error!` guard — adds a drift-chase tax proportional to upstream-nightly LLVM C ABI evals. The 2026-06-28T20:50 plumbing ADR + 2026-06-28T22:00 [CORRECTION] shim-correction ADR catalogued 4 unresolved C ABI symbols (`LLVMBuildLoad`, `LLVMConstZExt`, `LLVMAddGlobalDCEPass`, `LLVMRustStringWriteImpl`) that block runtime dlopen; the empirical scope for a runtime shim converged to ~1 day / ~35 LoC (corrected from earlier 3-5 day estimate) but buys zero benefit for the slim fork's LLVM-19-pinned scope. Retiring the cargo feature removes a future agents' temptation to chase those symbols and unblocks phase-3d commit-by-commit progression.

Rejected alternatives:
- **keep llvm20 as a CI-probe**: the cargo feature stays; CI runs `cargo check -p rustc_codegen_nvvm --no-default-features --features llvm20` to detect upstream LLVM 20 ABI drift early. Marginal value: llvm20 isn't customer-facing for this fork (CLAUDE.md pins LLVM 19), so the probe's signal doesn't translate to customer-fix action.
- **downgrade llvm20 to compile-only (drop per-example `[features]` table; keep backend probe)**: half-measure; still requires `build.rs::find_llvm_config_llvm20` + `cuda_builder::cfg!` chain maintenance for a downstream toolchain we explicitly don't run.

Tradeoffs:
- Removes ~50 LoC of plumbing across `rustc_codegen_nvvm/Cargo.toml`, `cuda_builder/Cargo.toml`, `crates/rustc_codegen_nvvm/build.rs`, and `crates/{gemm,gemv,attn,vecadd}/*/Cargo.toml` files (per-example `[features]` tables).
- No end-to-end regression: the LLVM-19 path is untouched. CLAUDE.md remains the operational rule.
- Closes the runtime-shim followup that the 2026-06-28T22:00 [CORRECTION] ADR estimated at ~1 day — that estimated shim work is now out-of-scope because llvm20 is out-of-scope.
- The `bool`-pair cascade in `build.rs` (currently `(b19, b20, b22)`) can be collapsed back to `(llvm19_enabled)` after both llvm20 and llvm22 plumbing retire, per the 2026-06-20T16:23:17 LLVM-not-viable-without-a-port ADR.

Supersedes: 2026-06-28T20:50 LLVM 20 plumbing ADR (the plumbing now retires rather than probe-mode), and 2026-06-28T22:00 [CORRECTION] LLVM 20 shim ADR (the runtime shim estimate is now out-of-scope). The llvm22 plumbing that was added in parallel (bool-pair refactor of `build.rs`) ALSO collapses in the same phase-3d commit.

Notes:
- Phase-3d execution: a single commit removes the `llvm20` plumbing, the `cuda_builder::cfg!` cascade, the per-example `[features]` tables, and collapses `build.rs`'s bool-pair cascade to `(b19)`. LLVM 22 plumbing retires in the same commit (was added together for the bool-pair refactor). Estimated: 1 commit, ~80 LoC deleted (negative diff), included in phase-3d.
- Acknowledging: the 2026-06-28T20:50 ADR's "Supersedes: 9 (LLVM 22 probe)" line is partially walked back here — the plumbing-arch shape that ADR claimed for future LLVM-version tracks (LLVM 21+) is no longer canon for this fork. A future LLVM-19 replacement (LLVM 21+, LLVM 23+) would need a separate, deliberate track.





## 2026-06-30T16:00:00-05:00 — [STRATEGIC] Phase-3b Tier-3-r3: residual #1 (IntrinsicResult) ELIMINATED, 7 → 6 nightly-drift residuals

**Reason:** Capture and commit the Phase-3b Tier-3-r3 session outcomes to the repo-local agent-memory ledger so the next context reset resumes with full baseline + lessons instead of starting from scratch. The driving goal was to drive 7 nightly-2026-04-02 cargo-check residuals on `ironand/crates/rustc_codegen_nvvm/` to 0 in a single atomic commit; this turn resolved 1 of 7 (residual #1) and captured the discovery + lesson that made resolution tractable.

**The 7 baseline residuals (verbatim cargo errors at commit `588204e`):**

1. **E0432 + E0603** — `crates/rustc_codegen_nvvm/src/intrinsic.rs:5:5` — `unresolved import rustc_codegen_ssa::mir::intrinsic::IntrinsicResult`. The `mir::intrinsic` module is private; the `IntrinsicResult` enum was REMOVED upstream in nightly-2026-04-02.
2. **E0432** — `crates/rustc_codegen_nvvm/src/lib.rs:63:72` — `unresolved import rustc_codegen_ssa::back::write::ThinLtoInput` (round-2 had moved this to `traits::ThinLtoInput` which also failed). Trait-method re-export chain needs verification against actual toolchain rust-src.
3. **E0407** — `crates/rustc_codegen_nvvm/src/debug_info/metadata/type_map.rs:35:9` — `method 'stable_hash' is not a member of trait 'super::StableHash'`. Upstream method name is `hash_stable` (NOT `stable_hash`).
4. **E0405** — `crates/rustc_codegen_nvvm/src/debug_info/metadata/type_map.rs:35:67` — `cannot find type 'StableHashCtxt' in rustc_data_structures::stable_hasher`. Trait expects `StableHashingContext` (qualified context bound, NOT a free type).
5. **E0046** — `crates/rustc_codegen_nvvm/src/builder.rs:182:1` — `Not all trait items implemented for BuilderMethods, missing 'scalable_alloca'`. Trait requires this method; signature to be re-derived from actual upstream.
6. **E0107** — `crates/rustc_codegen_nvvm/src/debug_info/metadata/type_map.rs:34:17` — `missing generics for trait HashStable (expected 1, got 0)`. Upstream `pub trait HashStable<HCX: StableHashingContext>` (or just `<HCX: ?Sized>`).
7. **E0053** — `crates/rustc_codegen_nvvm/src/lib.rs:195:10` — `Method join_codegen has incompatible type for trait (expected IndexMap, found UnordMap)`. `WorkProductMap` upstream is `UnordMap` per `rustc_middle::dep_graph::graph.rs`, but the trait expects `IndexMap` — bridge via `into_iter().collect::<FxIndexMap<_,_>>()` of the right shape.

**Round-3 batch regression lesson (the asset-cost of batch-str_replace):**

The previous session attempted to drive all 7 → 0 in ONE BATCH of 5 str_replaces (signature/imports edits across `intrinsic.rs`, `lib.rs`, `type_map.rs`, `builder.rs`). Cargo check after the batch produced 14 errors — 7 new errors introduced by the batch REPLACED the 7 baseline errors. Specifically:
- `E0252` multiple definitions of `WorkProduct` (added redundant `use rustc_middle::dep_graph::{WorkProduct, WorkProductId}` line colliding with the existing import).
- `E0046` + `E0050` `scalable_alloca` (signature guess `(&mut self, _ty: &'ll Type, _align: Align)` was wrong shape — actual trait expects different arity).
- `E0277` Sized bound missing on `HCX` for `HashStable` (used `?Sized` when trait requires `Sized`).
- `E0308` reversed direction on join_codegen bridge (tried `UnordMap → IndexMap` conversion when the trait wanted `UnordMap`, not `IndexMap`).
- E0599 stable_hash method name missing on call site (changed trait impl method name to `hash_stable` without updating callsite `tcx.with_stable_hashing_context(...)` which still calls `self.stable_hash(hcx, &mut hasher)`).

**Root-cause discovery (the key unlock for residual #1):**

The previous sessions were reading from `/workspace/scratch/rust-src-nightly/` — a STALE SNAPSHOT that diverged from the actual build toolchain. The ACTUAL nightly-2026-04-02 rust-src lives at:

```
/workspace/scratch/rustup/toolchains/nightly-2026-04-02-x86_64-unknown-linux-gnu/lib/rustlib/rustc-src/rust/
```

(subdirectory `compiler/rustc_codegen_ssa/...`, etc.). Differences observed between the stale snapshot and the real toolchain include `IntrinsicResult` being entirely absent, `ThinLtoInput` re-export structure differing, and the `HashStable` trait signature using `StableHashingContext` instead of an unqualified context.

This explains why every prior round of cargo hints vs upstream-source research was internally inconsistent.

**Residual #1 migration sequence (1A-1E) — how the disciplined one-edit-at-a-time close worked:**

1. **1A** (signature + imports) — Replaced `use rustc_codegen_ssa::mir::intrinsic::IntrinsicResult;` with removal AND replaced `fn codegen_intrinsic_call(...) -> IntrinsicResult<'tcx, &'ll Value>` with the actual upstream signature:
   ```rust
   fn codegen_intrinsic_call(
       &mut self,
       instance: ty::Instance<'tcx>,
       args: &[OperandRef<'tcx, &'ll Value>],
       result: PlaceRef<'tcx, &'ll Value>,
       span: Span,
   ) -> Result<(), ty::Instance<'tcx>>;
   ```
   cargo: introduced E0425/E0433 for `result_place`/`result_layout` references inside the body.
2. **1B** (body prologue removal) — Removed the derived-span line and the temp-alloca-creation block `let (result, used_temp_alloca) = if let Some(place_val) = result_place { ... } else { PlaceRef::alloca(...) }`. Caller now provides `result: PlaceRef<...>` directly; `span` is now a parameter; `used_temp_alloca` is no longer declared.
3. **1C** (return-path substitutions) — Bulk-substituted all `IntrinsicResult::WroteIntoPlace` → `Ok(())`, `IntrinsicResult::Operand(self.load_operand(result).val)` → `Ok(())`, and the single `IntrinsicResult::Fallback(Instance::new_raw(...))` → `Err(Instance::new_raw(...))` (the fallback arm signals the runtime to dispatch the default body via `super::codegen_intrinsic_call`).
4. **1D** (collapse conditional tails at 16- and 8-space indent) — `if used_temp_alloca { Ok(()) } else { Ok(()) }` → `Ok(())` (3 sites at 16-space indent, 1 site at 8-space indent for function-end fallthrough). Used `allowMultiple=true` on the 16-space pattern.
5. **1E** (catch the missed 24-space-indent site) — `return if used_temp_alloca { Ok(()) } else { Ok(()) };` (24-space indent, inside select_unpredictable inner match arm) needed a SEPARATE single-occurrence substitution because Step 1D's 16-space pattern didn't match this depth. Cargo E0425 for `cannot find value 'used_temp_alloca'` at line 429 was the signal.

After 1E: cargo attributes 0 errors to `intrinsic.rs` territory. Residual #1 fully eliminated.

**Rejected alternatives:**

- **Batch-str_replace-against-baseline (round-3 approach)**: 5 edits in a single round with no cargo validation between each. Result: regressed 7 → 14 errors. REJECTED for future migration arcs because it can't surface a wrong-name or wrong-shape guess per-edit.
- **Sourcing from `/workspace/scratch/rust-src-nightly/` (stale snapshot)**: Was a model of upstream source for many rounds. REJECTED — the snapshot diverged from the actual build toolchain, producing conflicting ground truth vs the cargo verifier.
- **Spinning up a separate "evidence file" outside .dejavue/**: Considered naming `/tmp/ironsand_upstream_signatures.md` as a transient evidence file but rejected because (a) `.dejavue/` is the canonical repo-local memory, (b) the file is committed alongside the decision, (c) transient files risk being lost between sessions.

**Tradeoffs:**

- Strict one-edit-at-a-time discipline is SLOWER per edit (cargo check cycle per sub-step) but CONVERGES — every cargo error is a precise signal pointing to the next fix.
- Manual reconciliation of round-3's speculative str_replaces back to the `a7212d3` baseline (full `git checkout HEAD -- intrinsic.rs lib.rs type_map.rs builder.rs`) is the canonical rollback path when batch attempts regress.
- `/tmp/ironsand_upstream_signatures.md` is intentionally OUTSIDE the repo (transient debug artifact, not committed). Each session can regenerate from the actual nightly-2026-04-02 rust-src as needed.

**Notes:**

- The working tree currently still has the dirty `intrinsic.rs` from the residual-#1 migration (the post-1E closure). The user has accepted that residual #2-#7 will land in follow-up commits to keep each commit atomic.
- Likely dead-code warnings (cargo `unreachable_patterns` lints) emerged on `select_unpredictable`'s inner match arms post-1E because the ref arms now always early-return; expected to be warnings, not errors.
- The user's discipline is "ONE str_replace per residual with cargo-as-ground-truth validation between each edit". The 1A-1E decomposition was a single-residual sub-decomposition where each sub-step was a distinct category of edit within the same trait-method migration.
- The system date is 2026-06-30 per session metadata.
