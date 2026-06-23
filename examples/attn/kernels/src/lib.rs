//! Attention kernels for ironsand — the prefill path of LLM inference.
//!
//! `mma_f16_tile`: Stage-0 spike proving `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`
//!   emits through rustc_codegen_nvvm/LLVM19.
//!
//! `flash_attn`: FlashAttention-2-style prefill kernel. Fixed Dh=128. One warp per block
//!   handles 16 query rows; loops over KV sequence in 16-wide tiles. Uses f16 mma.sync
//!   for both QK^T and PV matmuls with online softmax (row-max/sum via inline-asm shfl).
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]

mod flash_attn;
mod flash_attn_gqa;
mod mma_f16;

pub use crate::flash_attn::flash_attn;
pub use crate::flash_attn_gqa::flash_attn_gqa;
pub use crate::mma_f16::mma_f16_tile;
