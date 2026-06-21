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
use cuda_std::GpuFloat;
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

/// Unpack the 12-byte `scales` array (8 × 6-bit sub-scale + 8 × 6-bit sub-min,
/// bit-packed per llama.cpp) from 3 `u32`s into two `[u32; 8]` arrays. The
/// original [`scale_min`] does this byte-by-byte per sub-block; here we hoist
/// all 12 bytes into registers up front so the per-sub-block dot has no
/// scale-unpack latency.
///
/// Layout: bytes 0..7 hold the low halves of sc[0..3] and mn[0..3] (one 6-bit
/// value per byte, low 6 bits used). Bytes 8..11 hold the low 4 bits of
/// sc[4..7] and mn[4..7] (with the high 2 bits of those 8 values squeezed
/// into the top 2 bits of bytes 0..7). See `get_scale_min_k4` in
/// `ggml-quants.c` for the reference packer.
///
/// Per [`scale_min`], sub-block `j` (j ≥ 4) sources its high 2 bits from
/// byte `(j-4)` of the same array. So sc[4]'s high 2 bits come from byte 0
/// (NOT byte 3 — that's a common off-by-3 trap when reading the packed
/// layout at a glance), sc[5] from byte 1, sc[6] from byte 2, sc[7] from
/// byte 3. mn[4..7] likewise from bytes 4..7.
///
/// `s0` = bytes [0..4), `s1` = bytes [4..8), `s2` = bytes [8..12). Little-endian.
#[inline(always)]
fn unpack_q4k_scales(s0: u32, s1: u32, s2: u32) -> ([u32; 8], [u32; 8]) {
    let sc = [
        s0 & 0x3F,                                  // byte0 low 6
        (s0 >> 8) & 0x3F,                           // byte1 low 6
        (s0 >> 16) & 0x3F,                          // byte2 low 6
        (s0 >> 24) & 0x3F,                          // byte3 low 6
        (s2 & 0xF) | (((s0 >> 6) & 0x3) << 4),      // byte8 low 4 | byte0 high 2
        ((s2 >> 8) & 0xF) | (((s0 >> 14) & 0x3) << 4),   // byte9 low 4 | byte1 high 2
        ((s2 >> 16) & 0xF) | (((s0 >> 22) & 0x3) << 4),  // byte10 low 4 | byte2 high 2
        ((s2 >> 24) & 0xF) | (((s0 >> 30) & 0x3) << 4),  // byte11 low 4 | byte3 high 2
    ];
    let mn = [
        s1 & 0x3F,                                  // byte4 low 6
        (s1 >> 8) & 0x3F,                           // byte5 low 6
        (s1 >> 16) & 0x3F,                          // byte6 low 6
        (s1 >> 24) & 0x3F,                          // byte7 low 6
        ((s2 >> 4) & 0xF) | (((s1 >> 6) & 0x3) << 4),    // byte8 high 4 | byte4 high 2
        ((s2 >> 12) & 0xF) | (((s1 >> 14) & 0x3) << 4),  // byte9 high 4 | byte5 high 2
        ((s2 >> 20) & 0xF) | (((s1 >> 22) & 0x3) << 4),  // byte10 high 4 | byte6 high 2
        ((s2 >> 28) & 0xF) | (((s1 >> 30) & 0x3) << 4),  // byte11 high 4 | byte7 high 2
    ];
    (sc, mn)
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

/// Optimized Q4_K GEMV (v3): pair-of-sub-blocks loop + u32 scale reads + FMA.
///
/// Three changes on top of [`gemv_q4k_fast`]:
///
/// 1. **Per-group (64-weight) outer loop.** [`gemv_q4k_fast`] iterates per
///    32-weight sub-block and re-reads the same `qs` byte for sub=2g and
///    sub=2g+1 (since `qbase = bbase + 16 + (sub>>1) * 32` is identical for
///    the pair). The compiler likely can't CSE this through the runtime
///    `sub` value. v3 processes the two sub-blocks in a pair together:
///    one `qs` byte read → two nibbles → two FMAs. Halves the `qs` LDG count
///    vs fast.
///
/// 2. **u32 scale reads.** The 12-byte `scales` array is read as 3 `u32`s
///    once per super-block, then unpacked via [`unpack_q4k_scales`] into
///    two `[u32; 8]` arrays. Replaces 16 byte LDGs (8 sub-blocks × 2 bytes
///    from [`scale_min`]) with 3 u32 LDGs + register-resident unpack. The
///    compiler may have coalesced the byte reads, but the u32 path makes
///    the count explicit.
///
/// 3. **`f32::mul_add` FMA on the inner dot.** `b - m` is one FMA
///    (`d_eff·nib + (-m_eff)`), `acc` update is another (`(b-m)·x + acc`).
///    Two FMA instructions per sub-block, vs the MUL+SUB+MUL+ADD the source
///    would otherwise emit. The Q6_K fast experiment showed the compiler
///    already fuses simple MUL+ADD, but the affine `d·nib - m` form here
///    is less likely to be matched — measure, don't assume.
///
/// # Safety
/// As [`gemv_q4k_warp`].
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q4k_v3(
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
        let d = unsafe { cvt_f16(load_u16(aptr, bbase)) };
        let dmin = unsafe { cvt_f16(load_u16(aptr, bbase + 2)) };
        let scbase = bbase + 4;

        // 3 u32s cover all 12 scale bytes (144-byte blocks are 4-byte aligned,
        // so these LDG.U32 are aligned). Unpack into [u32; 8] arrays once;
        // the per-group loop indexes them by sub-block index.
        let s0 = unsafe { (aptr.add(scbase) as *const u32).read() };
        let s1 = unsafe { (aptr.add(scbase + 4) as *const u32).read() };
        let s2 = unsafe { (aptr.add(scbase + 8) as *const u32).read() };
        let (sc, mn) = unpack_q4k_scales(s0, s1, s2);

        let mut g = 0usize;
        while g < 4 {
            // Per group g ∈ [0,4): two sub-blocks (2g, 2g+1) share one qs byte.
            // sc[2g] / mn[2g] are the even-sub-block's scale/min;
            // sc[2g+1] / mn[2g+1] are the odd-sub-block's.
            let d_eff0 = d * sc[2 * g] as f32;
            let d_eff1 = d * sc[2 * g + 1] as f32;
            let m_eff0 = dmin * mn[2 * g] as f32;
            let m_eff1 = dmin * mn[2 * g + 1] as f32;
            let qbase = bbase + 16 + g * 32;
            let byte = unsafe { *aptr.add(qbase + lane as usize) };
            // Branchless nibble extract (was `if (sub & 1) == 1 { byte >> 4 }
            // else { byte & 0xF }` in the per-sub loop).
            let nib0 = (byte & 0xF) as f32;
            let nib1 = (byte >> 4) as f32;
            let gw0 = b * 256 + 2 * g * 32 + lane as usize;
            let gw1 = gw0 + 32;

            // (b - m) = d_eff·nib + (-m_eff) → one FMA.
            // acc = (b - m)·x + acc → one FMA.
            let bm0 = d_eff0.mul_add(nib0, -m_eff0);
            let bm1 = d_eff1.mul_add(nib1, -m_eff1);
            acc = bm0.mul_add(x[gw0], acc);
            acc = bm1.mul_add(x[gw1], acc);
            g += 1;
        }
        b += 1;
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}

/// Optimized Q4_K GEMV (v4): 2-way super-block unroll on top of v3.
///
/// [`gemv_q4k_v3`] processes one super-block per outer iteration. v4 processes
/// two, so the compiler can interleave block `b`'s FMAs with block `b+1`'s
/// `d`/`dmin`/scale/qs loads — Q4_K has heavier per-weight compute than Q6_K
/// (2 mults + 1 sub per weight, affine dequant), so the unroll gives more
/// independent work to fill the FMA pipe while waiting on memory. The tail
/// (odd `nb`) goes through the single-block path.
///
/// Register pressure: doubles the per-block state (2 × `[u32; 8]` for
/// `(sc, mn)`, 2 × `d`, 2 × `dmin`) in flight. With ~20-25 registers per
/// live state plus the inner-loop temporaries, the kernel stays within the
/// 64-register-per-lane budget; `nvvm` will spill to local memory if not.
///
/// # Safety
/// As [`gemv_q4k_warp`].
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q4k_v4(
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
    while b + 1 < nb {
        let bbase0 = row_base + b * BLK;
        let bbase1 = bbase0 + BLK;

        // Loads for both blocks (compiler is free to interleave with FMAs of b0).
        let d0 = unsafe { cvt_f16(load_u16(aptr, bbase0)) };
        let dmin0 = unsafe { cvt_f16(load_u16(aptr, bbase0 + 2)) };
        let s0a = unsafe { (aptr.add(bbase0 + 4) as *const u32).read() };
        let s0b = unsafe { (aptr.add(bbase0 + 8) as *const u32).read() };
        let s0c = unsafe { (aptr.add(bbase0 + 12) as *const u32).read() };
        let (sc0, mn0) = unpack_q4k_scales(s0a, s0b, s0c);

        let d1 = unsafe { cvt_f16(load_u16(aptr, bbase1)) };
        let dmin1 = unsafe { cvt_f16(load_u16(aptr, bbase1 + 2)) };
        let s1a = unsafe { (aptr.add(bbase1 + 4) as *const u32).read() };
        let s1b = unsafe { (aptr.add(bbase1 + 8) as *const u32).read() };
        let s1c = unsafe { (aptr.add(bbase1 + 12) as *const u32).read() };
        let (sc1, mn1) = unpack_q4k_scales(s1a, s1b, s1c);

        // Per-group: both blocks' group g in one go (compiler can schedule
        // the FMAs from b0 alongside the qs reads for b1).
        let mut g = 0usize;
        while g < 4 {
            // Block b, group g (sub 2g + sub 2g+1).
            let d_eff0a = d0 * sc0[2 * g] as f32;
            let d_eff0b = d0 * sc0[2 * g + 1] as f32;
            let m_eff0a = dmin0 * mn0[2 * g] as f32;
            let m_eff0b = dmin0 * mn0[2 * g + 1] as f32;
            let qbase0 = bbase0 + 16 + g * 32;
            let byte0 = unsafe { *aptr.add(qbase0 + lane as usize) };
            let nib0a = (byte0 & 0xF) as f32;
            let nib0b = (byte0 >> 4) as f32;
            let gw0a = b * 256 + 2 * g * 32 + lane as usize;
            let gw0b = gw0a + 32;
            acc = d_eff0a.mul_add(nib0a, -m_eff0a).mul_add(x[gw0a], acc);
            acc = d_eff0b.mul_add(nib0b, -m_eff0b).mul_add(x[gw0b], acc);

            // Block b+1, group g.
            let d_eff1a = d1 * sc1[2 * g] as f32;
            let d_eff1b = d1 * sc1[2 * g + 1] as f32;
            let m_eff1a = dmin1 * mn1[2 * g] as f32;
            let m_eff1b = dmin1 * mn1[2 * g + 1] as f32;
            let qbase1 = bbase1 + 16 + g * 32;
            let byte1 = unsafe { *aptr.add(qbase1 + lane as usize) };
            let nib1a = (byte1 & 0xF) as f32;
            let nib1b = (byte1 >> 4) as f32;
            let gw1a = (b + 1) * 256 + 2 * g * 32 + lane as usize;
            let gw1b = gw1a + 32;
            acc = d_eff1a.mul_add(nib1a, -m_eff1a).mul_add(x[gw1a], acc);
            acc = d_eff1b.mul_add(nib1b, -m_eff1b).mul_add(x[gw1b], acc);

            g += 1;
        }
        b += 2;
    }
    // Tail: one block left over (nb odd).
    if b < nb {
        let bbase = row_base + b * BLK;
        let d = unsafe { cvt_f16(load_u16(aptr, bbase)) };
        let dmin = unsafe { cvt_f16(load_u16(aptr, bbase + 2)) };
        let sa = unsafe { (aptr.add(bbase + 4) as *const u32).read() };
        let sb = unsafe { (aptr.add(bbase + 8) as *const u32).read() };
        let sc_u = unsafe { (aptr.add(bbase + 12) as *const u32).read() };
        let (sc, mn) = unpack_q4k_scales(sa, sb, sc_u);
        let mut g = 0usize;
        while g < 4 {
            let d_eff0 = d * sc[2 * g] as f32;
            let d_eff1 = d * sc[2 * g + 1] as f32;
            let m_eff0 = dmin * mn[2 * g] as f32;
            let m_eff1 = dmin * mn[2 * g + 1] as f32;
            let qbase = bbase + 16 + g * 32;
            let byte = unsafe { *aptr.add(qbase + lane as usize) };
            let nib0 = (byte & 0xF) as f32;
            let nib1 = (byte >> 4) as f32;
            let gw0 = b * 256 + 2 * g * 32 + lane as usize;
            let gw1 = gw0 + 32;
            acc = d_eff0.mul_add(nib0, -m_eff0).mul_add(x[gw0], acc);
            acc = d_eff1.mul_add(nib1, -m_eff1).mul_add(x[gw1], acc);
            g += 1;
        }
    }

    let sum = unsafe { warp_sum_f32(acc) };
    if lane == 0 {
        let e = unsafe { &mut *y.add(row) };
        *e = sum + beta * *e;
    }
}
