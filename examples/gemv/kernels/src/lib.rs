//! GEMV kernels for ironsand — the decode-phase hot path of LLM inference
//! (`y = alpha * A·x + beta * y`, with A an `m x k` weight and x a length-`k`
//! activation vector). The N=1 GEMM case cuBLAS bottoms out on, and the natural
//! first target for hand-written Rust kernels.
//!
//! f32 variants: `gemv_naive`, `gemv_block`, `gemv_warp`.
//! f16 variants (inference-relevant — weights are f16/bf16, half the bytes):
//! `gemv_f16_warp`, `gemv_f16_vec4`.
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]

mod gemv_block;
mod gemv_f16;
mod gemv_naive;
mod gemv_warp;

pub use crate::gemv_block::gemv_block;
pub use crate::gemv_f16::{gemv_f16_vec4, gemv_f16_warp};
pub use crate::gemv_naive::gemv_naive;
pub use crate::gemv_warp::gemv_warp;
