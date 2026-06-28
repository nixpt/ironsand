//! int8-quantized GEMV — the path toward zorro's quantized decode.
//!
//! Weights are symmetric-int8 quantized per row: `A[i,j] ≈ q[i,j] * scale_a[i]`
//! with `q` in `[-127, 127]`. int8 weights are ¼ the bytes of f32 and ½ of f16,
//! so on this bandwidth-bound kernel the decode latency drops accordingly.
//!
//! Two schemes:
//! - [`gemv_i8_warp`] — **W8A32**: int8 weights, f32 activations. Dequantize each
//!   weight to f32 in-register and accumulate in f32. No activation quantization.
//! - [`gemv_i8_dp4a`] — **W8A8**: int8 weights *and* int8 activations, using the
//!   `dp4a` 4-way int8→int32 dot-product instruction (the GPU analog of zorro's
//!   CPU int8 GEMV) with an i32 accumulator.
//!
//! Both load 4 int8 per 32-bit transaction (`u32`), so warps issue wide
//! coalesced reads. Requires `k % 4 == 0`.

use cuda_std::kernel;
use cuda_std::quant::{dp4a, warp_sum_f32, warp_sum_i32};
use cuda_std::thread;

const WARP: u32 = 32;

/// W8A32 GEMV: `y = scale_a[row] · (q·x) + beta·y`, int8 weights × f32 acts.
///
/// # Safety
/// `a` is `m*k` int8 weights as bytes; `scale_a` is length `m`; `x` is length
/// `k`; `y` is length `m`; `k % 4 == 0`; launch ≥ `m` warps, block % 32 == 0.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_i8_warp(
    a: &[u8],
    scale_a: &[f32],
    x: &[f32],
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
    let k4 = k / 4;
    let row_u32 = unsafe { a.as_ptr().add(row * k) as *const u32 };
    let mut acc = 0.0f32;
    let mut j = lane as usize;
    while j < k4 {
        let bytes = unsafe { *row_u32.add(j) }.to_le_bytes();
        let bx = j * 4;
        acc += (bytes[0] as i8 as f32) * x[bx];
        acc += (bytes[1] as i8 as f32) * x[bx + 1];
        acc += (bytes[2] as i8 as f32) * x[bx + 2];
        acc += (bytes[3] as i8 as f32) * x[bx + 3];
        j += WARP as usize;
    }
    let sum = unsafe { warp_sum_f32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = scale_a[row] * sum + beta * *e;
    }
}

/// W8A8 GEMV via dp4a: `y = scale_a[row]·scale_x·(q_a·q_x) + beta·y`.
///
/// # Safety
/// As [`gemv_i8_warp`], plus `xq` is length `k` int8 activations as bytes.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_i8_dp4a(
    a: &[u8],
    scale_a: &[f32],
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
    let k4 = k / 4;
    let arow = unsafe { a.as_ptr().add(row * k) as *const u32 };
    let xptr = unsafe { xq.as_ptr() as *const u32 };
    let mut acc: i32 = 0;
    let mut j = lane as usize;
    while j < k4 {
        let aw = unsafe { *arow.add(j) };
        let xw = unsafe { *xptr.add(j) };
        acc = unsafe { dp4a(aw, xw, acc) };
        j += WARP as usize;
    }
    let isum = unsafe { warp_sum_i32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = scale_a[row] * scale_x * (isum as f32) + beta * *e;
    }
}
