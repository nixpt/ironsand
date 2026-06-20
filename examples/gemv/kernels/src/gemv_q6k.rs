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
        let mut group = 0usize;
        while group < 2 {
            let qlb = bbase + group * 64;
            let qhb = bbase + 128 + group * 32;
            let scb = bbase + 192 + group * 8;
            let ql_l = unsafe { *aptr.add(qlb + l) } as i32;
            let ql_l32 = unsafe { *aptr.add(qlb + l + 32) } as i32;
            let qh = unsafe { *aptr.add(qhb + l) } as i32;

            let q1 = ((ql_l & 0xF) | ((qh & 3) << 4)) - 32;
            let q2 = ((ql_l32 & 0xF) | (((qh >> 2) & 3) << 4)) - 32;
            let q3 = ((ql_l >> 4) | (((qh >> 4) & 3) << 4)) - 32;
            let q4 = ((ql_l32 >> 4) | (((qh >> 6) & 3) << 4)) - 32;

            let s1 = unsafe { *aptr.add(scb + is) } as i8 as i32;
            let s2 = unsafe { *aptr.add(scb + is + 2) } as i8 as i32;
            let s3 = unsafe { *aptr.add(scb + is + 4) } as i8 as i32;
            let s4 = unsafe { *aptr.add(scb + is + 6) } as i8 as i32;

            let p = gw + group * 128 + l;
            bacc += (s1 * q1) as f32 * x[p];
            bacc += (s2 * q2) as f32 * x[p + 32];
            bacc += (s3 * q3) as f32 * x[p + 64];
            bacc += (s4 * q4) as f32 * x[p + 96];
            group += 1;
        }
        acc += d * bacc;
        b += 1;
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

/// `dp4a.s32.s32`: `c + Σ s8x4(a)·s8x4(b)` as i32. sm_61+.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn dp4a(a: u32, b: u32, c: i32) -> i32 {
    let d: i32;
    unsafe {
        asm!("dp4a.s32.s32 {d}, {a}, {b}, {c};",
            d = out(reg32) d, a = in(reg32) a, b = in(reg32) b, c = in(reg32) c)
    };
    d
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn dp4a(_a: u32, _b: u32, _c: i32) -> i32 {
    0
}

/// W6A8 Q6_K GEMV via dp4a (the mmvq-style integer-dot path).
///
/// Activations are pre-quantized to int8 (`xq`, per-vector `xscale`). Each lane
/// owns whole 16-weight sub-blocks (lane `l` → sub-blocks `l, l+32, …`); it
/// dequantizes the 16 weights to signed int8 `(q-32)` (no per-weight scale, since
/// a sub-block shares one int8 `scale`), packs them into 4 `u32`, and does 4
/// `dp4a` against the 16 contiguous int8 activations → `isum`. The sub-block then
/// contributes `d · xscale · scale · isum`. This replaces the W6A32 f32
/// dequant·multiply with one integer dot per 4 weights — the change that makes
/// the kernel competitive with zorro's mmvq.
///
/// RESULT: as written this is ~1.3-1.4x SLOWER than the W6A32 `gemv_q6k_warp`,
/// not faster. The lane-owns-sub-block layout scatters the ql/qh reads (16+16
/// bytes per lane across the block) and the per-weight unpack is heavy; that
/// overhead swamps dp4a's accumulate savings. Coalescing dominates dp4a here
/// (same lesson as the Q4_K restructure). A real mmvq-beating kernel needs
/// mmvq's register-tiled *coalesced* dp4a (consecutive threads read consecutive
/// bytes, activations gathered in weight-access order) — a faithful port, not a
/// naive dp4a. Kept as a correct integer-dot reference + a documented dead end.
///
/// # Safety
/// `k % 256 == 0`; `a` = `m*(k/256)` 210-byte blocks; `xq` = `k` int8 acts;
/// `y` = `m`; launch ≥ `m` warps, block % 32 == 0.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q6k_dp4a(
    a: &[u8],
    xq: &[u8],
    xscale: f32,
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
    let nsub = k / 16; // total 16-weight sub-blocks in a row
    let row_base = row * (k / 256) * BLK;
    let aptr = a.as_ptr();
    let xptr = xq.as_ptr();

    let mut acc = 0.0f32;
    let mut sbg = lane as usize; // global sub-block owned by this lane
    while sbg < nsub {
        let b = sbg / 16; // super-block
        let lsb = sbg % 16; // local sub-block 0..15
        let g = lsb / 8; // group 0/1
        let ls = lsb % 8;
        let lbase = (ls & 1) * 16;
        let qslot = ls >> 1; // 0..3
        let bbase = row_base + b * BLK;
        let d = unsafe { cvt_f16(load_u16(aptr, bbase + 208)) };
        let sc = unsafe { *aptr.add(bbase + 192 + lsb) } as i8 as i32;
        let qlb = bbase + g * 64;
        let qhb = bbase + 128 + g * 32;
        let gw = b * 256 + lsb * 16; // first weight (16 contiguous)

        let mut v = [0u32; 4];
        let mut i = 0usize;
        while i < 16 {
            let ll = lbase + i;
            let qh = unsafe { *aptr.add(qhb + ll) } as i32;
            let q6 = match qslot {
                0 => ((unsafe { *aptr.add(qlb + ll) } as i32) & 0xF) | ((qh & 3) << 4),
                1 => ((unsafe { *aptr.add(qlb + ll + 32) } as i32) & 0xF) | (((qh >> 2) & 3) << 4),
                2 => ((unsafe { *aptr.add(qlb + ll) } as i32) >> 4) | (((qh >> 4) & 3) << 4),
                _ => ((unsafe { *aptr.add(qlb + ll + 32) } as i32) >> 4) | (((qh >> 6) & 3) << 4),
            };
            let s8 = ((q6 - 32) as u32) & 0xFF;
            v[i >> 2] |= s8 << ((i & 3) * 8);
            i += 1;
        }

        let xptr32 = unsafe { xptr.add(gw) } as *const u32; // gw % 16 == 0
        let mut isum = 0i32;
        isum = unsafe { dp4a(v[0], *xptr32.add(0), isum) };
        isum = unsafe { dp4a(v[1], *xptr32.add(1), isum) };
        isum = unsafe { dp4a(v[2], *xptr32.add(2), isum) };
        isum = unsafe { dp4a(v[3], *xptr32.add(3), isum) };

        acc += d * xscale * (sc * isum) as f32;
        sbg += WARP as usize;
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}
