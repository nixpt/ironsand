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

