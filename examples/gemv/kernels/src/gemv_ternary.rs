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
use cuda_std::quant::{dp4a, spread, warp_sum_i32};
use cuda_std::thread;

const WARP: u32 = 32;

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

/// Optimized ternary GEMV via dp4a + branchless spread.
///
/// Replaces the scalar 16-iteration unpack with: per packed `u32` (16 codes),
/// [`spread`] each of the 4 bytes into 4 int8 lanes (codes `{0,1,2}`) and feed
/// `dp4a` against the int8 activations — 4 `dp4a` per word instead of 16
/// multiply-accumulates. The `-1` per weight is applied once at the end as
/// `Σ_j (c_j-1)·x_j = (Σ_j c_j·x_j) - Σ_j x_j`, where `x_sum = Σ_j x_q8[j]` is a
/// single host-provided scalar (the same for every row).
///
/// # Safety
/// As [`gemv_ternary_warp`], plus `x_sum` must equal the i32 sum of `xq` (as i8).
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_ternary_dp4a(
    w: &[u32],
    scale_w: &[f32],
    xq: &[u8],
    scale_x: f32,
    x_sum: i32,
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
    let kw = k / 16;
    let wrow = unsafe { w.as_ptr().add(row * kw) };
    let x32 = xq.as_ptr() as *const u32;

    let mut acc: i32 = 0;
    let mut t = lane as usize;
    while t < kw {
        let packed = unsafe { *wrow.add(t) };
        let xb = t * 4; // u32 index into the int8 activations
        acc = unsafe { dp4a(spread(packed & 0xFF), *x32.add(xb), acc) };
        acc = unsafe { dp4a(spread((packed >> 8) & 0xFF), *x32.add(xb + 1), acc) };
        acc = unsafe { dp4a(spread((packed >> 16) & 0xFF), *x32.add(xb + 2), acc) };
        acc = unsafe { dp4a(spread((packed >> 24) & 0xFF), *x32.add(xb + 3), acc) };
        t += WARP as usize;
    }

    let isum = unsafe { warp_sum_i32(acc) };
    if lane == 0 {
        let inner = isum - x_sum; // Σ(c-1)·x = Σc·x − Σx
        let e = unsafe { &mut *y.add(row) };
        *e = scale_w[row] * scale_x * (inner as f32) + beta * *e;
    }
}
