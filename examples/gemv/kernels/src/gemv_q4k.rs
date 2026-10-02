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

use core::mem::MaybeUninit;
use cuda_std::address_space;
use cuda_std::kernel;
use cuda_std::quant::{cvt_f16, dp4a, load_u16, scale_min, unpack_q4k_scales, warp_sum_f32};
use cuda_std::thread;

const WARP: u32 = 32;
const BLK: usize = 144; // bytes per Q4_K super-block

/// Threads per block for the fused mmq GEMM (one block per token).
const MMQ_BLK: usize = 256;
/// Max K (in-features) the shared activation buffer holds. Prefill cols = 2048.
const MMQ_MAXK: usize = 2048;

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
pub unsafe fn gemv_q4k_warp(a: &[u8], x: &[f32], y: *mut f32, m: usize, k: usize, beta: f32) {
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
        let nib = if (sub & 1) == 1 {
            byte >> 4
        } else {
            byte & 0xF
        };
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
pub unsafe fn gemv_q4k_fast(a: &[u8], x: &[f32], y: *mut f32, m: usize, k: usize, beta: f32) {
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
            let nib = if (sub & 1) == 1 {
                byte >> 4
            } else {
                byte & 0xF
            };
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
pub unsafe fn gemv_q4k_v3(a: &[u8], x: &[f32], y: *mut f32, m: usize, k: usize, beta: f32) {
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

/// Coalesced Q4_K vec_dot (mmvq-style) — W4A8.
///
/// Faithful port of llama.cpp `vec_dot_q4_K_q8_1_impl_vmmq`, mirroring the Q6_K
/// vecdot win. The whole warp processes one 256-weight super-block per step; the
/// 128-byte `qs` array is read fully coalesced — lane `l` reads the aligned `u32`
/// at `qs + l*4`, so 32 lanes cover all 128 bytes. Unlike Q6_K (210-byte blocks,
/// 2-aligned, needing u16-pair assembly), Q4_K blocks are 144 bytes = 4-aligned,
/// so the `u32` qs and scale loads are direct.
///
/// Each lane's `u32` holds 8 nibbles = 4 low-nibble weights (sub-block `2g`) and
/// 4 high-nibble weights (sub-block `2g+1`), with `g = lane/8`. The matching int8
/// activations are contiguous → one aligned `u32` load per sub-block. The affine
/// dequant `w = d·sc·q - dmin·mn` splits into two integer dots per sub-block:
///   `isum = dp4a(q, u)` → the `d·sc·Σ(q·x)` scale term, and
///   `asum = dp4a(1, u)` → the `dmin·mn·Σ(x)` constant-min term.
/// Lane accumulates `d·(sc·isum) - dmin·(mn·asum)` over its two sub-blocks across
/// all super-blocks; the per-vector `xscale` multiplies the warp-reduced sum.
/// (vs the dequant·dot [`gemv_q4k_fast`]/v3/v4 — those gather f32 acts per weight;
/// this is the coalesced integer-dot path that beat mmvq on Q6_K.)
///
/// # Safety
/// `k % 256 == 0`; `a` = `m·(k/256)` 144-byte blocks; `xq` = `k` int8; `y` = `m`.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_q4k_vecdot(
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
    let lane = (tid % WARP) as usize;
    if row >= m {
        return;
    }
    let nb = k / 256;
    let row_base = row * nb * BLK;
    let aptr = a.as_ptr();
    let x32 = xq.as_ptr() as *const u32;

    let gl = lane / 8; // 64-weight group 0..3 → sub-blocks 2g (lo) + 2g+1 (hi)
    let pl = lane % 8; // position within the group's 32 qs bytes
    let sub_lo = 2 * gl;
    let sub_hi = 2 * gl + 1;
    let byteoff = pl * 4; // first qs byte this lane owns (within the 32-byte group)
    let ones = 0x01010101u32;

    let mut acc = 0.0f32;
    let mut b = 0usize;
    while b < nb {
        let bbase = row_base + b * BLK;
        let d = unsafe { cvt_f16(load_u16(aptr, bbase)) };
        let dmin = unsafe { cvt_f16(load_u16(aptr, bbase + 2)) };

        // 12 scale bytes as 3 aligned u32 (144-byte blocks are 4-aligned). Broadcast.
        let s0 = unsafe { (aptr.add(bbase + 4) as *const u32).read() };
        let s1 = unsafe { (aptr.add(bbase + 8) as *const u32).read() };
        let s2 = unsafe { (aptr.add(bbase + 12) as *const u32).read() };
        let (sc, mn) = unpack_q4k_scales(s0, s1, s2);

        // Coalesced qs word: lane l reads bytes [gl*32+byteoff .. +4) of qs.
        let qoff = bbase + 16 + gl * 32 + byteoff;
        let qword = unsafe { (aptr.add(qoff) as *const u32).read() };
        let q_lo = qword & 0x0F0F0F0F; // 4 low nibbles  → sub_lo
        let q_hi = (qword >> 4) & 0x0F0F0F0F; // 4 high nibbles → sub_hi

        // Contiguous int8 activations for each sub-block (aligned u32 loads).
        let pos_lo = b * 256 + sub_lo * 32 + byteoff;
        let pos_hi = b * 256 + sub_hi * 32 + byteoff;
        let u_lo = unsafe { *x32.add(pos_lo / 4) };
        let u_hi = unsafe { *x32.add(pos_hi / 4) };

        // isum = Σ q·u (scale term); asum = Σ u (affine min term).
        let isum_lo = unsafe { dp4a(q_lo, u_lo, 0) };
        let asum_lo = unsafe { dp4a(ones, u_lo, 0) };
        let isum_hi = unsafe { dp4a(q_hi, u_hi, 0) };
        let asum_hi = unsafe { dp4a(ones, u_hi, 0) };

        let dsum = (sc[sub_lo] as i32 * isum_lo + sc[sub_hi] as i32 * isum_hi) as f32;
        let msum = (mn[sub_lo] as i32 * asum_lo + mn[sub_hi] as i32 * asum_hi) as f32;
        acc += d * dsum - dmin * msum;
        b += 1;
    }

    let sum = unsafe { warp_sum_f32(acc) } * xscale;
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
pub unsafe fn gemv_q4k_v4(a: &[u8], x: &[f32], y: *mut f32, m: usize, k: usize, beta: f32) {
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

/// Stage-1 fused Q4_K int8 mmq **batched GEMM** (dp4a inner) — the prefill lever.
///
/// Computes `Y[N×M] = X[N×K] · W[M×K]ᵀ` where `W` is Q4_K (resident bytes
/// `[M·(K/256)·144]` row-major) and `X` is f32 activations `[N×K]`. This is the
/// prefill projection (N≈512 tokens, K=cols=2048, M=rows∈{2048,8192}). It is the
/// FUSED design both foremen require — quantize-act + integer-dot + per-sub-block
/// `d·sc`/`dmin·mn` correction + output-scale ALL in one kernel (the unfused
/// cuBLAS-int8 path was a measured net-slower dead-end).
///
/// One block per token `n`: its threads (1) reduce `amax(X[n])` → per-token int8
/// scale, (2) quantize `X[n]` into shared `XQ` (int8), then (3) each thread sweeps
/// output features `m`, dequant-dotting W's super-blocks against `XQ` with `dp4a`
/// (`isum = Σ q·xq`) plus the affine min term (`asum = Σ xq`), exactly
/// `vec_dot_q4_K_q8_1` batched over N. Output is f32 here (Stage 1 = correctness);
/// Stage 2 swaps the dp4a inner dot for `mma.sync` int8 tensor cores + emits f16.
///
/// # Safety
/// `K % 256 == 0`, `K ≤ MMQ_MAXK`; `w` = `M·(K/256)·144` bytes; `x` = `N·K` f32;
/// `y` = `N·M` f32. Launch `<<<N, MMQ_BLK>>>`.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemm_q4k_mmq_dp4a(
    w: &[u8],
    x: &[f32],
    y: *mut f32,
    n: usize,
    mrows: usize,
    k: usize,
) {
    #[address_space(shared)]
    static mut XQ: [MaybeUninit<i8>; MMQ_MAXK] = [MaybeUninit::uninit(); MMQ_MAXK];
    #[address_space(shared)]
    static mut RED: [MaybeUninit<f32>; MMQ_BLK] = [MaybeUninit::uninit(); MMQ_BLK];
    #[address_space(shared)]
    static mut XSCALE: [MaybeUninit<f32>; 1] = [MaybeUninit::uninit(); 1];

    let tok = thread::block_idx_x() as usize;
    if tok >= n {
        return; // uniform across the block (one block ↔ one token)
    }
    let tid = thread::thread_idx_x() as usize;
    let xbase = tok * k;

    // (1) per-token amax → int8 scale.
    let mut local_max = 0.0f32;
    let mut i = tid;
    while i < k {
        let v = x[xbase + i].abs();
        if v > local_max {
            local_max = v;
        }
        i += MMQ_BLK;
    }
    unsafe { RED[tid].write(local_max) };
    thread::sync_threads();
    let mut stride = MMQ_BLK / 2;
    while stride >= 1 {
        if tid < stride {
            let a = unsafe { RED[tid].assume_init() };
            let b = unsafe { RED[tid + stride].assume_init() };
            unsafe { RED[tid].write(if a > b { a } else { b }) };
        }
        thread::sync_threads();
        stride >>= 1;
    }
    if tid == 0 {
        let amax = unsafe { RED[0].assume_init() };
        unsafe { XSCALE[0].write(if amax > 0.0 { amax / 127.0 } else { 1.0 }) };
    }
    thread::sync_threads();
    let xscale = unsafe { XSCALE[0].assume_init() };
    let inv = 1.0f32 / xscale;

    // (2) quantize X[tok] → shared int8.
    let mut i = tid;
    while i < k {
        let q = (x[xbase + i] * inv).round();
        let q = if q > 127.0 {
            127.0
        } else if q < -127.0 {
            -127.0
        } else {
            q
        };
        unsafe { XQ[i].write(q as i32 as i8) };
        i += MMQ_BLK;
    }
    thread::sync_threads();

    // (3) each thread sweeps output features m, fused dequant·dot vs XQ.
    let nb = k / 256;
    let wptr = w.as_ptr();
    let xqptr = core::ptr::addr_of!(XQ) as *const u8;
    let ones = 0x01010101u32;
    let mut m = tid;
    while m < mrows {
        let mut acc = 0.0f32;
        let mut b = 0usize;
        while b < nb {
            let wbase = (m * nb + b) * BLK;
            let d = unsafe { cvt_f16(load_u16(wptr, wbase)) };
            let dmin = unsafe { cvt_f16(load_u16(wptr, wbase + 2)) };
            let s0 = unsafe { (wptr.add(wbase + 4) as *const u32).read() };
            let s1 = unsafe { (wptr.add(wbase + 8) as *const u32).read() };
            let s2 = unsafe { (wptr.add(wbase + 12) as *const u32).read() };
            let (sc, mn) = unpack_q4k_scales(s0, s1, s2);

            let mut sub = 0usize;
            while sub < 8 {
                let g = sub >> 1;
                let qbase = wbase + 16 + g * 32;
                let kbase = b * 256 + sub * 32;
                let hi = (sub & 1) == 1;
                let mut isum = 0i32;
                let mut asum = 0i32;
                let mut t = 0usize;
                while t < 8 {
                    let raw = unsafe { (wptr.add(qbase + 4 * t) as *const u32).read() };
                    let wq4 = if hi {
                        (raw >> 4) & 0x0F0F0F0F
                    } else {
                        raw & 0x0F0F0F0F
                    };
                    let xq4 = unsafe { (xqptr.add(kbase + 4 * t) as *const u32).read() };
                    isum = unsafe { dp4a(wq4, xq4, isum) };
                    asum = unsafe { dp4a(ones, xq4, asum) };
                    t += 1;
                }
                acc += d * sc[sub] as f32 * isum as f32 - dmin * mn[sub] as f32 * asum as f32;
                sub += 1;
            }
            b += 1;
        }
        unsafe { *y.add(tok * mrows + m) = acc * xscale };
        m += MMQ_BLK;
    }
}
