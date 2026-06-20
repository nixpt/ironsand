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

