//! Q6_K GEMV — the GGUF 6-bit k-quant zorro uses for the lm_head + projections
//! (the fattest decode GEMVs: lm_head ~20% of a step).
//!
//! Q6_K packs 256 weights in 210-byte super-blocks (`block_q6_K`):
//! ```text
//!   ql    : [u8; 128]   low 4 bits of each 6-bit quant
//!   qh    : [u8;  64]   high 2 bits (4 per byte)
//!   scales: [i8;  16]   16 × int8 sub-block scales (one per 16 weights)
//!   d     : f16         super-block scale
//! ```
//! A weight's 6-bit value `q ∈ [0,63]` is `(ql_nibble) | (qh_2bits << 4)`;
//! dequant is `w = d · scale[sb] · (q - 32)`. The ql/qh interleaving follows
//! llama.cpp's `dequantize_row_q6_K`: 2 groups of 128, lane `l ∈ [0,32)` owns
//! weights `l, l+32, l+64, l+96` per group (coalesced ql/qh reads). 6.5625
//! bits/weight. Requires `k % 256 == 0`.

use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::warp;
use cuda_std::GpuFloat;
#[cfg(target_os = "cuda")]
use core::arch::asm;

const WARP: u32 = 32;
const BLK: usize = 210; // bytes per Q6_K super-block

#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn cvt_f16(bits: u16) -> f32 {
    let o: f32;
    unsafe { asm!("cvt.f32.f16 {o}, {i};", o = out(reg32) o, i = in(reg16) bits) };
    o
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn cvt_f16(_bits: u16) -> f32 {
    0.0
}

#[inline(always)]
unsafe fn warp_sum_f32(mut v: f32) -> f32 {
    let mut off = WARP / 2;
    while off >= 1 {
        let (bits, _) = unsafe { warp::warp_shuffle_xor(u32::MAX, v.to_bits(), off, WARP) };
        v += f32::from_bits(bits);
        off >>= 1;
    }
    v
}

/// Per-group inner dot. Each lane owns 4 weights (`l, l+32, l+64, l+96` of the
/// 128-weight group) and reads 2 ql bytes + 1 qh byte + 4 scale bytes. The 4
/// `s*q` products are FMA'd into `bacc` over `gw..gw+128`.
///
/// Scale reads: lanes 0..15 (`is=0`) take even-indexed bytes of the 8-byte half;
/// lanes 16..31 (`is=1`) take odd-indexed bytes. For group 0 the half is
/// `scb..scb+7`; for group 1 it is `scb..scb+7` of the upper half
/// (i.e. block's `bbase+200..bbase+207`).
///
/// # Safety
/// `qlb/qhb/scb` point to the group's ql/qh/scales; `l ∈ [0,32)`;
/// `x` has ≥ `gw+128` entries.
#[inline(always)]
unsafe fn q6k_group_dot(
    aptr: *const u8,
    qlb: usize,
    qhb: usize,
    scb: usize,
    l: usize,
    is: usize,
    gw: usize,
    x: &[f32],
) -> f32 {
    let ql_l = unsafe { *aptr.add(qlb + l) } as i32;
    let ql_l32 = unsafe { *aptr.add(qlb + l + 32) } as i32;
    let qh = unsafe { *aptr.add(qhb + l) } as i32;
    // 6-bit codes: low nibble of ql + (qh's 2-bit slice << 4) − 32 (zero-centered).
    let q1 = ((ql_l & 0xF) | ((qh & 3) << 4)) - 32;
    let q2 = ((ql_l32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
    let q3 = ((ql_l >> 4) | (((qh >> 4) & 3) << 4)) - 32;
    let q4 = ((ql_l32 >> 4) | (((qh >> 6) & 3) << 4)) - 32;
    let s1 = unsafe { *aptr.add(scb + is) } as i8 as i32;
    let s2 = unsafe { *aptr.add(scb + is + 2) } as i8 as i32;
    let s3 = unsafe { *aptr.add(scb + is + 4) } as i8 as i32;
    let s4 = unsafe { *aptr.add(scb + is + 6) } as i8 as i32;
    let p = gw + l;
    // `f32::mul_add` lowers to PTX `fma.rn.f32` (one instruction, vs MUL+ADD
    // emitting as two if the optimizer misses it). The 4 FMAs are independent
    // over `p` and can issue in parallel.
    let mut bacc = 0.0f32;
    bacc = ((s1 * q1) as f32).mul_add(x[p], bacc);
    bacc = ((s2 * q2) as f32).mul_add(x[p + 32], bacc);
    bacc = ((s3 * q3) as f32).mul_add(x[p + 64], bacc);
    bacc = ((s4 * q4) as f32).mul_add(x[p + 96], bacc);
    bacc
}

/// Warp-per-row Q6_K GEMV. `a` holds `m * (k/256)` super-blocks (210 bytes each);
/// `x` is length `k`; `y` is length `m`. Only lanes 0..32 carry weight (one
/// quarter of a 128-group each); the per-block `d` factors out of the inner sum.
///
/// # Safety
/// `k % 256 == 0`; buffers sized as above; launch ≥ `m` warps, block % 32 == 0.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q6k_warp(
    a: &[u8],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    beta: f32,
) {
    let tid = thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x();
    let row = (tid / WARP) as usize;
    let l = (tid % WARP) as usize; // 0..32
    if row >= m {
        return;
    }
    let nb = k / 256;
    let row_base = row * nb * BLK;
    let aptr = a.as_ptr();

    let mut acc = 0.0f32;
    let mut b = 0usize;
    while b < nb {
        let bbase = row_base + b * BLK;
        let d = unsafe { cvt_f16(load_u16(aptr, bbase + 208)) };
        let gw = b * 256;
        let is = l >> 4; // l/16 → 0 or 1

        let mut bacc = 0.0f32;
        // Group 0: weights [gw, gw+128) → ql[0..64], qh[0..32], scales[0..8).
        bacc += unsafe { q6k_group_dot(aptr, bbase, bbase + 128, bbase + 192, l, is, gw, x) };
        // Group 1: weights [gw+128, gw+256) → ql[64..128], qh[32..64], scales[8..16).
        bacc += unsafe { q6k_group_dot(aptr, bbase + 64, bbase + 160, bbase + 200, l, is, gw + 128, x) };
        acc += d * bacc;
        b += 1;
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if l == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}

/// Optimized Q6_K GEMV: `mul_add` FMA on the inner dot and the outer
/// `d·bacc + acc`, plus a 2-way manual unroll of the super-block loop so the
/// compiler can interleave block `b`'s FMAs with block `b+1`'s ql/qh/scale
/// loads (each block has ~7 byte-reads/cycle, 2 blocks gives 14 = 2 FMA issues'
/// worth of latency to hide).
///
/// The per-block dot logic is in [`q6k_group_dot`]; called here in
/// block-group order (g0, g1 for block `b`, then g0, g1 for block `b+1`) so
/// both blocks' loads can be in flight before any FMA consumes them. The tail
/// super-block (when `nb` is odd) goes through the single-block path.
///
/// **Status: tried, no win.** PTX confirms `gemv_q6k_warp` already emits 10
/// `fma.rn.f32` (the compiler fuses MUL+ADD); this kernel emits 28 (2-way
/// unroll + the `mul_add` chain). Across all 5 bench shapes wall time is
/// ~equal to or slightly slower than `warp` (best +5% on 4096², worst −7%
/// on 32000x4096) because Q6_K is memory-bound at ~380 GB/s — the extra FMAs
/// just sit waiting on loads. Kept in tree as a documented experiment; see
/// the dejavue decision "Q6_K fast (mul_add FMA + 2-way super-block unroll)
/// gives no speedup over Q6_K warp".
///
/// # Safety
/// As [`gemv_q6k_warp`].
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q6k_fast(
    a: &[u8],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    beta: f32,
) {
    let tid = thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x();
    let row = (tid / WARP) as usize;
    let l = (tid % WARP) as usize;
    if row >= m {
        return;
    }
    let nb = k / 256;
    let row_base = row * nb * BLK;
    let aptr = a.as_ptr();
    let is = l >> 4;

    let mut acc = 0.0f32;
    let mut b = 0usize;
    // 2-way unroll. `nb` for our shapes (4096/256=16, 11008/256=43 odd, 12288/256=48,
    // 32000/256=125 odd) is mostly even; the tail falls through to the single path.
    while b + 1 < nb {
        let bbase0 = row_base + b * BLK;
        let bbase1 = bbase0 + BLK;
        let d0 = unsafe { cvt_f16(load_u16(aptr, bbase0 + 208)) };
        let d1 = unsafe { cvt_f16(load_u16(aptr, bbase1 + 208)) };
        let gw0 = b * 256;
        let gw1 = gw0 + 256;

        // Block b: g0 + g1 → bacc0.
        let mut bacc0 = 0.0f32;
        bacc0 += unsafe { q6k_group_dot(aptr, bbase0, bbase0 + 128, bbase0 + 192, l, is, gw0, x) };
        bacc0 += unsafe { q6k_group_dot(aptr, bbase0 + 64, bbase0 + 160, bbase0 + 200, l, is, gw0 + 128, x) };
        // Block b+1: g0 + g1 → bacc1. (Group dots read independently — the compiler
        // is free to reorder the ql/qh/scale loads across both blocks.)
        let mut bacc1 = 0.0f32;
        bacc1 += unsafe { q6k_group_dot(aptr, bbase1, bbase1 + 128, bbase1 + 192, l, is, gw1, x) };
        bacc1 += unsafe { q6k_group_dot(aptr, bbase1 + 64, bbase1 + 160, bbase1 + 200, l, is, gw1 + 128, x) };

        acc = d0.mul_add(bacc0, acc);
        acc = d1.mul_add(bacc1, acc);
        b += 2;
    }
    // Tail: one block left over (nb odd) — use the simple path.
    if b < nb {
        let bbase = row_base + b * BLK;
        let d = unsafe { cvt_f16(load_u16(aptr, bbase + 208)) };
        let gw = b * 256;
        let mut bacc = 0.0f32;
        bacc += unsafe { q6k_group_dot(aptr, bbase, bbase + 128, bbase + 192, l, is, gw, x) };
        bacc += unsafe { q6k_group_dot(aptr, bbase + 64, bbase + 160, bbase + 200, l, is, gw + 128, x) };
        acc = d.mul_add(bacc, acc);
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if l == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}

#[inline(always)]
unsafe fn load_u16(p: *const u8, off: usize) -> u16 {
    let lo = unsafe { *p.add(off) } as u16;
    let hi = unsafe { *p.add(off + 1) } as u16;
    lo | (hi << 8)
}
