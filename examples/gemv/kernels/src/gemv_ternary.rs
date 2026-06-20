//! Ternary (i2_s-style) GEMV — zorro's BitNet decode format.
//!
//! BitNet b1.58 weights are ternary `{-1, 0, +1}` with a per-row scale; with
//! int8-quantized activations the projection is
//! `y[i] = scale_w[i] · scale_x · Σ_j (w_tern[i,j] · x_q8[j])`,
//! and the inner product needs no multiplies — each ternary weight just adds,
//! subtracts, or skips its activation (zorro does this on CPU with
//! `_mm256_sign_epi8`).
//!
//! Packing: each weight is a 2-bit code `w + 1 ∈ {0,1,2}` (so `signed = code-1`),
//! 16 codes per `u32`. That is **2 bits/weight** — ¼ the bytes of int8 and 1/16
//! of f32. GEMV is bandwidth bound, so in principle that is another 4× over
//! int8 — but at this density the 2-bit unpack can shift the bottleneck from
//! memory to ALU, which is exactly why zorro needed SIMD sign tricks on CPU.
//!
//! Requires `k % 16 == 0`.

use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::warp;

const WARP: u32 = 32;

#[inline(always)]
unsafe fn warp_sum_i32(mut v: i32) -> i32 {
    let mut off = WARP / 2;
    while off >= 1 {
        let (bits, _) = unsafe { warp::warp_shuffle_xor(u32::MAX, v as u32, off, WARP) };
        v += bits as i32;
        off >>= 1;
    }
    v
}

/// Warp-per-row ternary GEMV. `w` holds `m * (k/16)` packed `u32` words (16
/// ternary codes each); `xq` is `k` int8 activations; `scale_w` is length `m`.
///
/// # Safety
/// `k % 16 == 0`; buffers sized as above; `y` length `m`; launch ≥ `m` warps,
/// block % 32 == 0.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_ternary_warp(
    w: &[u32],
    scale_w: &[f32],
    xq: &[u8],
    scale_x: f32,
    y: *mut f32,
    m: usize,
    k: usize,
    beta: f32,
) {
    let tid = thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x();
    let row = (tid / WARP) as usize;
    let lane = tid % WARP;
    if row >= m {
        return;
    }
    let kw = k / 16; // u32 words per row
    let wrow = unsafe { w.as_ptr().add(row * kw) };
    let xptr = xq.as_ptr();

    let mut acc: i32 = 0;
    let mut t = lane as usize;
    while t < kw {
        let mut packed = unsafe { *wrow.add(t) };
        let base = t * 16;
        // Unpack 16 ternary codes; `signed = code - 1 ∈ {-1, 0, 1}`.
        let mut b = 0usize;
        while b < 16 {
            let signed = (packed & 0x3) as i32 - 1;
            let xi = unsafe { *xptr.add(base + b) } as i8 as i32;
            acc += signed * xi;
            packed >>= 2;
            b += 1;
        }
        t += WARP as usize;
    }

    let isum = unsafe { warp_sum_i32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = scale_w[row] * scale_x * (isum as f32) + beta * *e;
    }
}
