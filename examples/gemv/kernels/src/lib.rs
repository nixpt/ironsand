//! GEMV kernels for ironsand — the decode-phase hot path of LLM inference
//! (`y = alpha * A·x + beta * y`, with A an `m x k` weight and x a length-`k`
//! activation vector). The N=1 GEMM case cuBLAS bottoms out on, and the natural
//! first target for hand-written Rust kernels.
//!
//! f32 variants: `gemv_naive`, `gemv_block`, `gemv_warp`.
//! f16 variants (inference-relevant — weights are f16/bf16, half the bytes):
//! `gemv_f16_warp`, `gemv_f16_vec4`.
//! int8 variants (quantized decode — ¼ the bytes of f32): `gemv_i8_warp`
//! (W8A32), `gemv_i8_dp4a` (W8A8, dp4a integer dot).
//! ternary variant (zorro BitNet i2_s — 2 bits/weight): `gemv_ternary_warp`.
//! Q4_K variant (GGUF 4-bit k-quant): `gemv_q4k_warp`/`gemv_q4k_fast`.
//! Q6_K variant (GGUF 6-bit k-quant, lm_head): `gemv_q6k_warp`.
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]

mod gemv_block;
mod gemv_f16;
mod gemv_i8;
mod gemv_naive;
mod gemv_q4k;
mod gemv_q6k;
mod gemv_ternary;
mod gemv_warp;

pub use crate::gemv_block::gemv_block;
pub use crate::gemv_f16::{gemv_f16_vec4, gemv_f16_warp};
pub use crate::gemv_i8::{gemv_i8_dp4a, gemv_i8_warp};
pub use crate::gemv_naive::gemv_naive;
pub use crate::gemv_q4k::{gemv_q4k_fast, gemv_q4k_warp};
pub use crate::gemv_q6k::{gemv_q6k_dp4a, gemv_q6k_warp};
pub use crate::gemv_ternary::{gemv_ternary_dp4a, gemv_ternary_warp};
pub use crate::gemv_warp::gemv_warp;
