# Handoff

Updated: 2026-06-20T16:23:30-05:00

## Summary
Fork stood up, slimmed to zorro inference scope, and proven: Rust->PTX via LLVM 19 runs on the GPU (gemm naive+tiled, async stream overlap). Build needs LLVM_CONFIG_19=/workspace/scratch/llvm19/bin/llvm-config + CUDA_PATH=/opt/cuda.

## Next Steps
- First net-new zorro kernel experiment: write a Rust attention or fused-GEMV kernel, benchmark vs blastoff/cuBLAS and vs zorro's CUDA-C++ kernels
- Optional: create GitHub remote (nixpt/ironsand) and push
- Optional: incremental crate renames to ironsand_* with re-export shims

## Boot Instructions
Read `.dejavue/handoff.md`, `.dejavue/state.md`, `.dejavue/decisions.md`, and `.dejavue/timeline.jsonl` before making changes.
