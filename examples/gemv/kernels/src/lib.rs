//! GEMV kernels for ironsand — the decode-phase hot path of LLM inference
//! (`y = alpha * A·x + beta * y`, with A an `m x k` row-major weight and x a
//! length-`k` activation vector). This is the N=1 GEMM case that cuBLAS bottoms
//! out on, so it is the natural first target for hand-written Rust kernels.

mod gemv_block;
mod gemv_naive;
mod gemv_warp;

pub use crate::gemv_block::gemv_block;
pub use crate::gemv_naive::gemv_naive;
pub use crate::gemv_warp::gemv_warp;
