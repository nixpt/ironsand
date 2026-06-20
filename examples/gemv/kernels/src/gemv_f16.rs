//! f16-weight GEMV: `y = alpha * A·x + beta * y` with A stored as f16 (passed as
//! raw `u16` bits), x and y in f32, accumulating in f32.
//!
//! This is the inference-relevant case: LLM weights are f16/bf16, so A streams
//! at *half* the bytes of the f32 path. GEMV is memory-bandwidth bound, so half
//! the bytes ≈ half the latency per projection — directly the decode win.
//!
//! f16→f32 uses the hardware `cvt.f32.f16` instruction via inline PTX (the
//! `half` crate's `to_f32` lowers to a software bit-twiddling path instead).

use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::warp;
#[cfg(target_os = "cuda")]
use core::arch::asm;

const WARP: u32 = 32;

/// Hardware f16→f32 conversion. `bits` holds the raw IEEE half bit pattern.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn cvt(bits: u16) -> f32 {
    let o: f32;
    // dest is .f32 (reg32 = a 32-bit %r register holding the float bits), src is
    // .f16 (reg16 = a 16-bit %rs register).
    unsafe { asm!("cvt.f32.f16 {o}, {i};", o = out(reg32) o, i = in(reg16) bits) };
    o
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn cvt(_bits: u16) -> f32 {
    0.0
}

/// Butterfly all-reduce of an f32 across the warp (over the value's bits, since
/// the shuffle intrinsics are integer-typed).
#[inline(always)]
unsafe fn warp_sum(mut v: f32) -> f32 {
    let mut off = WARP / 2;
    while off >= 1 {
        let (bits, _) = unsafe { warp::warp_shuffle_xor(u32::MAX, v.to_bits(), off, WARP) };
        v += f32::from_bits(bits);
        off >>= 1;
    }
    v
}

/// Warp-per-row f16 GEMV, scalar 16-bit loads.
///
/// # Safety
/// `a` is `m*k` f16 values as `u16` bits; `x` is length `k`; `y` is length `m`;
/// launch ≥ `m` warps with `block_dim_x` a multiple of 32.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_f16_warp(
    a: &[u16],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let tid = thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x();
    let row = (tid / WARP) as usize;
    let lane = tid % WARP;
    if row >= m {
        return;
    }
    let base = row * k;
    let mut partial = 0.0f32;
    let mut i = lane as usize;
    while i < k {
        partial += unsafe { cvt(a[base + i]) } * x[i];
        i += WARP as usize;
    }
    let sum = unsafe { warp_sum(partial) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = alpha * sum + beta * *e;
    }
}

/// Warp-per-row f16 GEMV, vectorized 64-bit loads (4 f16 per load).
///
/// Each lane loads a `u64` (4 packed f16) per step, so a warp pulls 32×8 = 256
/// contiguous bytes per step — wide, fully-coalesced transactions. Requires
/// `k % 4 == 0` and 8-byte row alignment (true for `cudaMalloc` + `k % 4 == 0`).
///
/// # Safety
/// Same as [`gemv_f16_warp`], plus `k % 4 == 0`.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_f16_vec4(
    a: &[u16],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let tid = thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x();
    let row = (tid / WARP) as usize;
    let lane = tid % WARP;
    if row >= m {
        return;
    }
    let k4 = k / 4;
    let row_u64 = unsafe { a.as_ptr().add(row * k) as *const u64 };
    let mut partial = 0.0f32;
    let mut j = lane as usize;
    while j < k4 {
        let packed = unsafe { *row_u64.add(j) };
        let bx = j * 4;
        partial += unsafe { cvt((packed & 0xffff) as u16) } * x[bx];
        partial += unsafe { cvt(((packed >> 16) & 0xffff) as u16) } * x[bx + 1];
        partial += unsafe { cvt(((packed >> 32) & 0xffff) as u16) } * x[bx + 2];
        partial += unsafe { cvt(((packed >> 48) & 0xffff) as u16) } * x[bx + 3];
        j += WARP as usize;
    }
    let sum = unsafe { warp_sum(partial) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = alpha * sum + beta * *e;
    }
}
