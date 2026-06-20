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

