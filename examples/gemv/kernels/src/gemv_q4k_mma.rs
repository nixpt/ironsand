//! Stage-2: fused Q4_K int8 **tensor-core** mmq batched GEMM (`mma.sync`).
//!
//! `Y[N×M] = X[N×K] · dequant(W)ᵀ`, W = Q4_K. The Stage-1 dp4a scaffold proved
//! the fused math; this swaps the inner dot for `mma.sync.m16n8k32.s8` int8
//! tensor cores (Stage-0 proved the emit) — the only path to the ~60 TFLOP/s the
//! cuBLAS-f16 prefill bar runs at.
//!
//! Two kernels (llama's mmq design):
//!   1. [`quant_act_q8`] — per-token amax→int8 activation quant + per-32 block
//!      sum (`bsum`, for the affine `dmin·mn` min term, llama's `ds8.y`).
//!   2. [`gemm_q4k_mma`] — warp owns one 16×8 output tile; K is walked in 32-wide
//!      steps (= one Q4_K sub-block = one `mma.k32`). Per step: load Xq (A frag) +
//!      W nibbles (B frag), `mma` → int32, then apply that sub-block's `d·sc`
//!      scale and subtract `dmin·mn·bsum` in f32. Per-token `xscale` at the epilogue.
//!
//! Fragment layout = canonical Ampere+ (`grp = lane/4`, `lane2 = lane%4`):
//!   A[16×32]: a0/a2 row=grp (k 0-15/16-31), a1/a3 row=grp+8.
//!   B[8×32]:  n=grp; b0 k 0-15, b1 k 16-31.
//!   C[16×8]:  c0,c1 (row grp, col lane2·2+{0,1}); c2,c3 (row grp+8, …).

use core::mem::MaybeUninit;
use cuda_std::address_space;
use cuda_std::kernel;
use cuda_std::quant::{cvt_f16, f16_bits, load_u16, unpack_q4k_scales};
use cuda_std::thread;

const QBLK: usize = 256; // threads/block for the activation-quant kernel
const QMAXK: usize = 2048; // shared activation buffer (prefill cols = 2048)
const BLK: usize = 144; // bytes per Q4_K super-block

/// `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32` — one int8 tensor-core tile.
/// `a` = 4 packed-s8 regs, `b` = 2, accumulator `c` = 4 s32; returns D (= C + A·Bᵀ).
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn mma_s8(a: [u32; 4], b: [u32; 2], c: [i32; 4]) -> [i32; 4] {
    let (mut d0, mut d1, mut d2, mut d3) = (c[0], c[1], c[2], c[3]);
    unsafe {            core::arch::asm!(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{0}, {1}, {2}, {3}}}, {{{4}, {5}, {6}, {7}}}, {{{8}, {9}}}, {{{0}, {1}, {2}, {3}}};",
            inout(reg32) d0,
            inout(reg32) d1,
            inout(reg32) d2,
            inout(reg32) d3,
            in(reg32) a[0],
            in(reg32) a[1],
            in(reg32) a[2],
            in(reg32) a[3],
            in(reg32) b[0],
            in(reg32) b[1],
        );
    }
    [d0, d1, d2, d3]
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn mma_s8(_a: [u32; 4], _b: [u32; 2], c: [i32; 4]) -> [i32; 4] {
    c
}

/// Per-token int8 activation quant + per-32-block sum, for [`gemm_q4k_mma`].
/// `x` = `[N×K]` f32; writes `xq` = `[N×K]` int8, `xscale` = `[N]`,
/// `bsum` = `[N×(K/32)]` i32 (Σ of the int8 in each 32-block). Launch `<<<N, QBLK>>>`.
///
/// # Safety
/// `K ≤ QMAXK`, `K % 32 == 0`; buffers sized as above.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn quant_act_q8(
    x: &[f32],
    xq: *mut i8,
    xscale: *mut f32,
    bsum: *mut i32,
    n: usize,
    k: usize,
) {
    #[address_space(shared)]
    static mut XQ: [MaybeUninit<i8>; QMAXK] = [MaybeUninit::uninit(); QMAXK];
    #[address_space(shared)]
    static mut RED: [MaybeUninit<f32>; QBLK] = [MaybeUninit::uninit(); QBLK];
    #[address_space(shared)]
    static mut SC: [MaybeUninit<f32>; 1] = [MaybeUninit::uninit(); 1];

    let tok = thread::block_idx_x() as usize;
    if tok >= n {
        return;
    }
    let tid = thread::thread_idx_x() as usize;
    let xbase = tok * k;

    let mut local_max = 0.0f32;
    let mut i = tid;
    while i < k {
        let v = x[xbase + i].abs();
        if v > local_max {
            local_max = v;
        }
        i += QBLK;
    }
    unsafe { RED[tid].write(local_max) };
    thread::sync_threads();
    let mut stride = QBLK / 2;
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
        let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        unsafe { SC[0].write(scale) };
        unsafe { *xscale.add(tok) = scale };
    }
    thread::sync_threads();
    let inv = 1.0f32 / unsafe { SC[0].assume_init() };

    // Quantize into shared + global.
    let mut i = tid;
    while i < k {
        let q = (x[xbase + i] * inv).round();
        let q = if q > 127.0 {
            127.0
        } else if q < -127.0 {
            -127.0
        } else {
            q
        } as i32 as i8;
        unsafe { XQ[i].write(q) };
        unsafe { *xq.add(xbase + i) = q };
        i += QBLK;
    }
    thread::sync_threads();

    // Per-32-block sums (the affine `dmin·mn` min term needs Σ of int8 acts).
    let nsub = k / 32;
    let mut s = tid;
    while s < nsub {
        let mut acc = 0i32;
        let mut w = 0usize;
        while w < 32 {
            acc += unsafe { XQ[s * 32 + w].assume_init() } as i32;
            w += 1;
        }
        unsafe { *bsum.add(tok * nsub + s) = acc };
        s += QBLK;
    }
}

/// Load this lane's B fragment (2 regs) of W nibbles for output features
/// `j = tj + (lane/4)`, Q4_K super-block `b`, sub-block `subin`.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn load_b_nibbles(
    wptr: *const u8,
    jrow_base: usize,
    b: usize,
    subin: usize,
    lane2: usize,
) -> [u32; 2] {
    let qbase = jrow_base + b * BLK + 16 + (subin >> 1) * 32;
    let r0 = unsafe { (wptr.add(qbase + lane2 * 4) as *const u32).read() };
    let r1 = unsafe { (wptr.add(qbase + lane2 * 4 + 16) as *const u32).read() };
    if subin & 1 == 1 {
        [(r0 >> 4) & 0x0F0F0F0F, (r1 >> 4) & 0x0F0F0F0F]
    } else {
        [r0 & 0x0F0F0F0F, r1 & 0x0F0F0F0F]
    }
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn load_b_nibbles(_w: *const u8, _j: usize, _b: usize, _s: usize, _l: usize) -> [u32; 2] {
    [0, 0]
}

/// Tensor-core Q4_K int8 mmq GEMM. `Y[N×M] = X[N×K]·dequant(W)ᵀ`. Inputs are the
/// pre-quantized activations from [`quant_act_q8`]. Output is **f16** (`y` as u16
/// bits) so zorro's gate→silu→down chain stays fused. One warp = one 16×8 tile;
/// launch `<<<(N/16)*(M/8), 32>>>`.
///
/// # Safety
/// `N % 16 == 0`, `M % 8 == 0`, `K % 32 == 0`; `w` = Q4_K `[M·(K/256)·144]`;
/// `xq` = `[N×K]` i8; `xscale` = `[N]`; `bsum` = `[N×(K/32)]`; `y` = `[N×M]` u16.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemm_q4k_mma(
    w: &[u8],
    xq: &[u8],
    xscale: &[f32],
    bsum: &[i32],
    y: *mut u16,
    n: usize,
    mrows: usize,
    k: usize,
) {
    let tile = thread::block_idx_x() as usize;
    let tiles_m = mrows / 8;
    let ti = (tile / tiles_m) * 16; // token base
    let tj = (tile % tiles_m) * 8; // output-feature base
    if ti >= n {
        return;
    }
    let lane = thread::thread_idx_x() as usize;
    let grp = lane / 4; // 0..7
    let lane2 = lane % 4; // 0..3

    let nb = k / 256;
    let nsub = k / 32;
    let wptr = w.as_ptr();
    let xqptr = xq.as_ptr();

    // C-fragment output coords this lane owns: rows i0/i1, cols j0/j1.
    let i0 = ti + grp;
    let i1 = ti + grp + 8;
    let j0 = tj + lane2 * 2;
    let j1 = tj + lane2 * 2 + 1;
    let j0_base = j0 * nb * BLK;
    let j1_base = j1 * nb * BLK;
    // B-fragment outfeat this lane LOADS (= tj+grp; distinct from j0/j1 by design).
    let jb_base = (tj + grp) * nb * BLK;

    let (mut f0, mut f1, mut f2, mut f3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);

    let mut sub = 0usize;
    while sub < nsub {
        let b = sub / 8;
        let subin = sub % 8;
        let kbase = sub * 32;

        // A fragment: Xq for tokens i0 (rows grp) and i1 (rows grp+8).
        let a = [
            unsafe { (xqptr.add(i0 * k + kbase + lane2 * 4) as *const u32).read() },
            unsafe { (xqptr.add(i1 * k + kbase + lane2 * 4) as *const u32).read() },
            unsafe { (xqptr.add(i0 * k + kbase + lane2 * 4 + 16) as *const u32).read() },
            unsafe { (xqptr.add(i1 * k + kbase + lane2 * 4 + 16) as *const u32).read() },
        ];
        // B fragment: W nibbles for outfeat (tj+grp).
        let bfrag = unsafe { load_b_nibbles(wptr, jb_base, b, subin, lane2) };

        let d = unsafe { mma_s8(a, bfrag, [0, 0, 0, 0]) };

        // Per-output-column (j0, j1) Q4_K scale/min from W's super-block b, subin.
        let d0w = unsafe { cvt_f16(load_u16(wptr, j0_base + b * BLK)) };
        let dmin0 = unsafe { cvt_f16(load_u16(wptr, j0_base + b * BLK + 2)) };
        let (sc0, mn0) = unsafe {
            unpack_q4k_scales(
                (wptr.add(j0_base + b * BLK + 4) as *const u32).read(),
                (wptr.add(j0_base + b * BLK + 8) as *const u32).read(),
                (wptr.add(j0_base + b * BLK + 12) as *const u32).read(),
            )
        };
        let d1w = unsafe { cvt_f16(load_u16(wptr, j1_base + b * BLK)) };
        let dmin1 = unsafe { cvt_f16(load_u16(wptr, j1_base + b * BLK + 2)) };
        let (sc1, mn1) = unsafe {
            unpack_q4k_scales(
                (wptr.add(j1_base + b * BLK + 4) as *const u32).read(),
                (wptr.add(j1_base + b * BLK + 8) as *const u32).read(),
                (wptr.add(j1_base + b * BLK + 12) as *const u32).read(),
            )
        };

        let s0 = d0w * sc0[subin] as f32;
        let m0 = dmin0 * mn0[subin] as f32;
        let s1 = d1w * sc1[subin] as f32;
        let m1 = dmin1 * mn1[subin] as f32;
        let bs0 = unsafe { *bsum.get_unchecked(i0 * nsub + sub) } as f32;
        let bs1 = unsafe { *bsum.get_unchecked(i1 * nsub + sub) } as f32;

        f0 += s0 * d[0] as f32 - m0 * bs0; // (i0, j0)
        f1 += s1 * d[1] as f32 - m1 * bs0; // (i0, j1)
        f2 += s0 * d[2] as f32 - m0 * bs1; // (i1, j0)
        f3 += s1 * d[3] as f32 - m1 * bs1; // (i1, j1)

        sub += 1;
    }

    let xs0 = xscale[i0];
    let xs1 = xscale[i1];
    unsafe {
        *y.add(i0 * mrows + j0) = f16_bits(f0 * xs0);
        *y.add(i0 * mrows + j1) = f16_bits(f1 * xs0);
        *y.add(i1 * mrows + j0) = f16_bits(f2 * xs1);
        *y.add(i1 * mrows + j1) = f16_bits(f3 * xs1);
    }
}
