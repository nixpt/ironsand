//! Q4_K GEMV — the GGUF k-quant 4-bit format zorro decodes.
//!
//! Q4_K packs weights in 256-element super-blocks (`block_q4_K`, 144 bytes):
//! ```text
//!   d     : f16            super-block scale (scale of the sub-scales)
//!   dmin  : f16            super-block min   (scale of the sub-mins)
//!   scales: [u8; 12]       8 × 6-bit sub-scale + 8 × 6-bit sub-min, bit-packed
//!   qs    : [u8; 128]      256 × 4-bit quants (low/high nibbles per 64 group)
//! ```
//! Each 32-weight sub-block `sb` has a 6-bit scale `sc` and 6-bit min `mn`;
//! dequant is affine: `w = d·sc·q - dmin·mn`, `q ∈ [0,15]`. That is 4.5
//! bits/weight (between int8 and ternary), but the per-sub-block scale unpack +
//! affine dequant is heavier ALU than the symmetric int8/ternary paths.
//!
//! Layout note: this is byte-faithful to llama.cpp's `block_q4_K`
//! (`get_scale_min_k4` + the low/high-nibble `qs` order). Requires `k % 256 == 0`.

use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::warp;
#[cfg(target_os = "cuda")]
use core::arch::asm;

const WARP: u32 = 32;
const BLK: usize = 144; // bytes per Q4_K super-block

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
unsafe fn load_u16(p: *const u8, off: usize) -> u16 {
    let lo = unsafe { *p.add(off) } as u16;
    let hi = unsafe { *p.add(off + 1) } as u16;
    lo | (hi << 8)
}

/// llama.cpp `get_scale_min_k4`: unpack sub-block `j`'s 6-bit scale and min from
/// the 12-byte `scales` array at `sc` (a pointer to `scales[0]`).
#[inline(always)]
unsafe fn scale_min(j: usize, sc: *const u8) -> (u32, u32) {
    unsafe {
        if j < 4 {
            ((*sc.add(j) & 63) as u32, (*sc.add(j + 4) & 63) as u32)
        } else {
            let d = ((*sc.add(j + 4) & 0xF) | ((*sc.add(j - 4) >> 6) << 4)) as u32;
            let m = ((*sc.add(j + 4) >> 4) | ((*sc.add(j) >> 6) << 4)) as u32;
            (d, m)
        }
    }
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

/// Warp-per-row Q4_K GEMV. `a` holds `m * (k/256)` super-blocks (144 bytes each);
/// `x` is length `k`; `y` is length `m`.
///
/// Lane `l` owns weight `l` of every 32-weight sub-block, so the nibble and
/// activation reads are coalesced across the warp while the per-sub-block header
/// (d/dmin/scales) loads broadcast (same address for all lanes).
///
/// # Safety
/// `k % 256 == 0`; buffers sized as above; launch ≥ `m` warps, block % 32 == 0.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q4k_warp(
    a: &[u8],
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
    let nb = k / 256;
    let row_base = row * nb * BLK;
    let nsub = k / 32;
    let aptr = a.as_ptr();

    let mut acc = 0.0f32;
    let mut s = 0usize;
    while s < nsub {
        let gw0 = s * 32; // first weight of this sub-block
        let b = gw0 / 256; // super-block
        let sub = (gw0 / 32) & 7; // sub-block 0..7
        let bbase = row_base + b * BLK;

        let d = unsafe { cvt_f16(load_u16(aptr, bbase)) };
        let dmin = unsafe { cvt_f16(load_u16(aptr, bbase + 2)) };
        let (sc, mn) = unsafe { scale_min(sub, aptr.add(bbase + 4)) };
        let d_eff = d * sc as f32;
        let m_eff = dmin * mn as f32;

        let g = sub >> 1; // 64-weight group
        let qbase = bbase + 16 + g * 32;
        let byte = unsafe { *aptr.add(qbase + lane as usize) };
        let nib = if (sub & 1) == 1 { byte >> 4 } else { byte & 0xF };
        let w = d_eff * (nib as f32) - m_eff;
        acc += w * x[gw0 + lane as usize];

        s += 1;
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}

/// Optimized Q4_K GEMV: decode the super-block `d`/`dmin` once per 256 weights.
///
/// [`gemv_q4k_warp`] keeps the coalesced lane=weight layout (good), but its loop
/// is per sub-block, so it re-`cvt_f16`s the super-block `d` and `dmin` on every
/// one of the 8 sub-blocks — 8× redundant. Here the outer loop is per
/// super-block: decode `d`/`dmin` once, then an inner loop over the 8 sub-blocks
/// does only the per-sub-block `get_scale_min_k4` + the coalesced nibble read.
/// Same memory access pattern, fewer f16 conversions.
///
/// (An earlier attempt that made each lane own whole sub-blocks amortized the
/// header decode but scattered the nibble reads and ran ~1.6× *slower* —
/// coalescing dominates the header ALU here.)
///
/// # Safety
/// As [`gemv_q4k_warp`].
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q4k_fast(
    a: &[u8],
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
    let nb = k / 256;
    let row_base = row * nb * BLK;
    let aptr = a.as_ptr();

    let mut acc = 0.0f32;
    let mut b = 0usize;
    while b < nb {
        let bbase = row_base + b * BLK;
        // Decoded once per super-block (8× fewer than the per-sub-block kernel).
        let d = unsafe { cvt_f16(load_u16(aptr, bbase)) };
        let dmin = unsafe { cvt_f16(load_u16(aptr, bbase + 2)) };
        let scbase = bbase + 4;

        let mut sub = 0usize;
        while sub < 8 {
            let (sc, mn) = unsafe { scale_min(sub, aptr.add(scbase)) };
            let d_eff = d * sc as f32;
            let m_eff = dmin * mn as f32;
            let g = sub >> 1;
            let qbase = bbase + 16 + g * 32;
            let byte = unsafe { *aptr.add(qbase + lane as usize) }; // coalesced
            let nib = if (sub & 1) == 1 { byte >> 4 } else { byte & 0xF };
            let gw = b * 256 + sub * 32 + lane as usize;
            acc += (d_eff * (nib as f32) - m_eff) * x[gw];
            sub += 1;
        }
        b += 1;
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}
