//! GEMV kernels for ironsand — the decode-phase hot path of LLM inference
//! (`y = alpha * A·x + beta * y`, with A an `m x k` row-major weight and x a
//! length-`k` activation vector). This is the N=1 GEMM case that cuBLAS bottoms
//! out on, so it is the natural first target for hand-written Rust kernels.
//!
//! `gemv_warp` (warp-shuffle reduction) is kept as source but NOT compiled: its
//! shuffle intrinsics crash libnvvm during PTX generation (see the dejavue
//! `warp_shuffle` trap). `gemv_block` is the portable coalesced variant.

mod gemv_block;
mod gemv_naive;
// mod gemv_warp; // disabled: warp_shuffle_xor SIGSEGVs libnvvm — see trap.

pub use crate::gemv_block::gemv_block;
pub use crate::gemv_naive::gemv_naive;
