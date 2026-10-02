//! FlashAttention-2-style prefill v4: O in per-thread f32 registers (no O_SMEM).
//!
//! `O[L, Dh] = softmax(Q[L,Dh] · K[S,Dh]^T / √Dh) · V[S, Dh]`
//!
//! Fixed: Dh=128, Br=16 (query rows/block), Bc=16 (KV columns/step). 4 warps per block.
//! Grid: `(⌈L/16⌉,)` blocks × 128 threads. L and S must be multiples of 16.
//!
//! ## v4 over v3
//!   - **O in registers**: O_SMEM (8 KB f32) replaced by per-thread `o_g[8]`/`o_g8[8]`.
//!     Layout: `o_g[tt*2+r]` = O[grp, dh_base+tt*8+l2*2+r] for tt=0..3, r=0..1.
//!     Rescale/normalize/accumulate are pure register ops — no smem traffic.
//!   - **Smem: 20 KB → 12 KB**: enables 4 blocks/SM → 16 warps/SM (50% occ, was 25%).
//!   - **Final write**: scattered per-thread stores (16 f16 cells/thread) replacing coalesced
//!     smem→global pass. Eliminates 8 KB smem + one sync_threads().
//!
//! ## Parallelism strategy
//!
//! **4 warps per block**, smem 12 KB:
//!   - All 4 warps compute the same S[16,16] = Q·K^T (redundant but avoids S_smem).
//!   - Warp w owns the Dh-slice [w*32, (w+1)*32) of O and 4 PV n-tiles starting at t=w*4.
//!
//! Occupancy: 48 KB smem / 12 KB per block = 4 blocks per SM → 16 warps per SM.
//!
//! ## Fragment layout (Ampere+, grp=lane/4, l2=lane%4, warp_id=tid/32)
//! QK^T A: ldmatrix.x4 → [a0..a3] from Q_smem[l%16, kb+(l/16)*8].
//! QK^T B: ldmatrix.x2 (no trans, ×2/k-step) from K_smem row-major.
//! PV A: register-packed P values.
//! PV B: ldmatrix.x2 (×1/n-tile) from V_T_smem col-major B layout.

#[cfg(target_os = "cuda")]
use core::arch::asm;
use core::mem::MaybeUninit;
use core::ptr::{addr_of, addr_of_mut};
use cuda_std::GpuFloat;
use cuda_std::address_space;
use cuda_std::kernel;
use cuda_std::thread;

use crate::mma_f16::mma_f16;

const BR: usize = 16;
const BC: usize = 16;
const DH: usize = 128;
const NWARPS: usize = 4;
const BLOCK_THREADS: usize = NWARPS * 32;
const DH_PER_WARP: usize = DH / NWARPS; // = 32
const VTILES_PER_WARP: usize = (DH / 8) / NWARPS; // = 4  (DH/8 = 16 total n-tiles)

// ── Inline-asm helpers ──────────────────────────────────────────────────────

#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn cvt_f32_f16(v: f32) -> u16 {
    let o: u16;
    unsafe { asm!("cvt.rn.f16.f32 {o}, {i};", o = out(reg16) o, i = in(reg32) v) };
    o
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn cvt_f32_f16(_v: f32) -> u16 {
    0
}

/// Pack two f32→f16 into one u32: lo bits [0:15], hi bits [16:31].
#[inline(always)]
unsafe fn pack_f16(lo: f32, hi: f32) -> u32 {
    (unsafe { cvt_f32_f16(lo) } as u32) | ((unsafe { cvt_f32_f16(hi) } as u32) << 16)
}

#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn shfl_xor_f32(v: f32, offset: u32) -> f32 {
    let bits: u32;
    unsafe {
        asm!(
            "shfl.sync.bfly.b32 {0}, {1}, {2}, 0x1f, 0xffffffff;",
            out(reg32) bits,
            in(reg32) v.to_bits(),
            in(reg32) offset,
        );
    }
    f32::from_bits(bits)
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn shfl_xor_f32(v: f32, _offset: u32) -> f32 {
    v
}

/// Max across 4-thread group (XOR 2, then XOR 1 — stays within group).
#[inline(always)]
unsafe fn group_max(v: f32) -> f32 {
    let v = {
        let w = unsafe { shfl_xor_f32(v, 2) };
        if w > v { w } else { v }
    };
    let w = unsafe { shfl_xor_f32(v, 1) };
    if w > v { w } else { v }
}

/// Sum across 4-thread group.
#[inline(always)]
unsafe fn group_sum(mut v: f32) -> f32 {
    v += unsafe { shfl_xor_f32(v, 2) };
    v += unsafe { shfl_xor_f32(v, 1) };
    v
}

/// Cooperative warp load of 4×(m8n8) f16 submatrices from shared memory into A-fragment regs.
///
/// In PTX .address_size 64 mode, `ldmatrix` accepts a 64-bit generic address pointing to
/// shared memory — no `cvta.to.shared` needed.
/// Lane l provides Q_smem[l%16, kb + (l/16)*8] (16-byte row within the 4×8×8 tile).
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ldmatrix_a4(addr: u64) -> [u32; 4] {
    let (mut a0, mut a1, mut a2, mut a3): (u32, u32, u32, u32);
    unsafe {
        asm!(
            "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {{{0},{1},{2},{3}}}, [{4}];",
            out(reg32) a0,
            out(reg32) a1,
            out(reg32) a2,
            out(reg32) a3,
            in(reg64) addr,
        );
    }
    [a0, a1, a2, a3]
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn ldmatrix_a4(_addr: u64) -> [u32; 4] {
    [0; 4]
}

/// Load 4×u32 (128 bits = 8×u16) from global memory.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ld_global_v4(addr: u64) -> [u32; 4] {
    let (mut r0, mut r1, mut r2, mut r3): (u32, u32, u32, u32);
    unsafe {
        asm!(
            "ld.global.v4.b32 {{{0},{1},{2},{3}}}, [{4}];",
            out(reg32) r0,
            out(reg32) r1,
            out(reg32) r2,
            out(reg32) r3,
            in(reg64) addr,
        );
    }
    [r0, r1, r2, r3]
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn ld_global_v4(_addr: u64) -> [u32; 4] {
    [0; 4]
}

/// Store 4×u32 (128 bits) to shared memory (address must be 16-byte aligned).
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn st_shared_v4(addr: u64, data: [u32; 4]) {
    unsafe {
        asm!(
            "st.shared.v4.b32 [{0}], {{{1},{2},{3},{4}}};",
            in(reg64) addr,
            in(reg32) data[0],
            in(reg32) data[1],
            in(reg32) data[2],
            in(reg32) data[3],
        );
    }
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn st_shared_v4(_addr: u64, _data: [u32; 4]) {}

/// Cooperative warp load of 2×(m8n8) f16 submatrices from shared memory, **no transpose**.
/// Works for both K and V_T B-fragments — source is row-major in both cases.
///
/// For K B-fragment (n-tile j=0): lane l provides K_smem[(l&7)*DH + kb + (l&8)].
///   After ldmatrix: thread (grp,l2) gets b0=K[grp, kb+l2*2..+1], b1=K[grp, kb+8+l2*2..+1].
/// For K B-fragment (n-tile j=1): lane l provides K_smem[((l&7)+8)*DH + kb + (l&8)].
/// For V_T B-fragment: lane l provides V_T_smem[(t*8+(l&7))*BC + (l&8)].
/// Lanes 16-31 are ignored by the hardware (provide any valid smem pointer).
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ldmatrix_b2(addr: u64) -> [u32; 2] {
    let (mut b0, mut b1): (u32, u32);
    unsafe {
        asm!(
            "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {{{0},{1}}}, [{2}];",
            out(reg32) b0,
            out(reg32) b1,
            in(reg64) addr,
        );
    }
    [b0, b1]
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn ldmatrix_b2(_addr: u64) -> [u32; 2] {
    [0; 2]
}

// ── Kernel ──────────────────────────────────────────────────────────────────

/// FlashAttention-2 prefill v4+: 4-warp DH-split, O in per-thread registers, multi-head.
/// `O[H,L,Dh] = softmax(Q·K^T/√Dh)·V`. f16 in/out.
/// Dh=128, L%16==0, S%16==0. Layout: [H, L, Dh] (head-major).
/// Launch `<<<(ceil(L/16), H), 128>>>` (2D grid over query tiles × heads).
///
/// q_head_stride = l_seq * Dh, kv_head_stride = s_seq * Dh (pre-computed on host).
/// Head-specific pointers are computed once in the prologue so hot loops are identical
/// to the single-head v4 kernel (no runtime offset in the per-tile load address chains).
///
/// # Safety
/// `L % 16 == 0`, `S % 16 == 0`, `Dh == 128`.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn flash_attn(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    o: *mut u16,
    l_seq: usize,
    s_seq: usize,
    q_head_stride: usize,
    kv_head_stride: usize,
) {
    #[address_space(shared)]
    static mut Q_SMEM: [MaybeUninit<u16>; BR * DH] = [MaybeUninit::uninit(); BR * DH];
    #[address_space(shared)]
    static mut K_SMEM: [MaybeUninit<u16>; BC * DH] = [MaybeUninit::uninit(); BC * DH];
    #[address_space(shared)]
    static mut V_T_SMEM: [MaybeUninit<u16>; DH * BC] = [MaybeUninit::uninit(); DH * BC];
    // O_SMEM removed in v4 — O lives in per-thread registers (o_g / o_g8).
    // Smem: 3×4 KB = 12 KB (was 20 KB with O_SMEM) → 4 blocks/SM, 16 warps/SM.

    // Raw smem pointers — avoids Rust 2024 ban on &T/&mut T to static mut.
    let q_smem = addr_of_mut!(Q_SMEM) as *mut MaybeUninit<u16>;
    let k_smem = addr_of_mut!(K_SMEM) as *mut MaybeUninit<u16>;
    let vt_smem = addr_of_mut!(V_T_SMEM) as *mut MaybeUninit<u16>;

    let tid = thread::thread_idx_x() as usize;
    let warp_id = tid / 32; // 0..3
    let lane = tid % 32; // lane within warp (0..31)
    let grp = lane / 4; // groupID 0..7
    let l2 = lane % 4; // threadID-in-group 0..3
    let h_idx = thread::block_idx_y() as usize; // head index
    let qi_tile = thread::block_idx_x() as usize;
    let query_base = qi_tile * BR;
    if query_base >= l_seq {
        return;
    }
    // Compute head-specific base pointers ONCE (prologue, not hot loops).
    // All hot-loop address computation is then identical to single-head v4 —
    // the compiler can precompute per-thread smem address constants as before.
    let q_head = unsafe { q.as_ptr().add(h_idx * q_head_stride) };
    let k_head = unsafe { k.as_ptr().add(h_idx * kv_head_stride) };
    let v_head = unsafe { v.as_ptr().add(h_idx * kv_head_stride) };
    let o_head = unsafe { o.add(h_idx * q_head_stride) };

    let scale = 1.0f32 / (DH as f32).sqrt();

    // Each warp owns the Dh-slice [warp_id*DH_PER_WARP, (warp_id+1)*DH_PER_WARP).
    let dh_base = warp_id * DH_PER_WARP; // 0, 32, 64, or 96
    let t_base = warp_id * VTILES_PER_WARP; // 0, 4, 8, or 12

    // ── Load Q tile (2 vectorized 128-bit loads per thread) ─────────────────────
    // Thread tid handles 2 chunks of 8 consecutive u16s (one chunk per BR half):
    //   chunk_a: Q[(query_base + tid/16), (tid%16)*8 .. +7]  (rows 0..7)
    //   chunk_b: Q[(query_base + (tid+BLOCK_THREADS)/16), ((tid+BLOCK_THREADS)%16)*8 .. +7] (rows 8..15)
    // Address is always 16-byte aligned: col = (tid%16)*8 u16 = (tid%16)*16 bytes.
    {
        let row_a = tid >> 4;
        let col_a = (tid & 15) << 3;
        let v4a = unsafe { ld_global_v4(q_head.add((query_base + row_a) * DH + col_a) as u64) };
        unsafe { st_shared_v4(q_smem.add(row_a * DH + col_a) as u64, v4a) };

        let e_b = tid + BLOCK_THREADS;
        let row_b = e_b >> 4;
        let col_b = (e_b & 15) << 3;
        let v4b = unsafe { ld_global_v4(q_head.add((query_base + row_b) * DH + col_b) as u64) };
        unsafe { st_shared_v4(q_smem.add(row_b * DH + col_b) as u64, v4b) };
    }
    thread::sync_threads();

    // Per-thread O accumulators in f32 registers (no O_SMEM).
    // o_g[tt*2+r]  = O[grp,   dh_base + tt*8 + l2*2 + r]  (tt=0..3, r=0..1)
    // o_g8[tt*2+r] = O[grp+8, dh_base + tt*8 + l2*2 + r]
    let mut o_g: [f32; 8] = [0.0; 8];
    let mut o_g8: [f32; 8] = [0.0; 8];

    let mut m_g = f32::NEG_INFINITY;
    let mut m_g8 = f32::NEG_INFINITY;
    let mut l_g = 0.0f32;
    let mut l_g8 = 0.0f32;

    // ── KV loop ─────────────────────────────────────────────────────────────
    let mut kv_tile = 0usize;
    while kv_tile < s_seq / BC {
        let kv_base = kv_tile * BC;

        // ── Load K [BC, DH] — 2 vectorized 128-bit loads per thread ─────────────
        {
            let row_a = tid >> 4;
            let col_a = (tid & 15) << 3;
            let v4a = unsafe { ld_global_v4(k_head.add((kv_base + row_a) * DH + col_a) as u64) };
            unsafe { st_shared_v4(k_smem.add(row_a * DH + col_a) as u64, v4a) };

            let e_b = tid + BLOCK_THREADS;
            let row_b = e_b >> 4;
            let col_b = (e_b & 15) << 3;
            let v4b = unsafe { ld_global_v4(k_head.add((kv_base + row_b) * DH + col_b) as u64) };
            unsafe { st_shared_v4(k_smem.add(row_b * DH + col_b) as u64, v4b) };
        }
        // ── Load V^T into V_T_SMEM [DH, BC] — scalar scatter (transpose on load) ──
        // V_T_SMEM[dh, bc] = V[bc, dh]. Scalar loop with XOR swizzle to eliminate 8-way
        // bank conflicts. Swizzle by multiple of 8 to preserve ldmatrix alignment:
        // bc_swizzled = bc ^ (((dh >> 3) & 1) * 8). (Phase-8, ±8 per dh level.)
        {
            let mut e = tid;
            while e < BC * DH {
                let bc = e / DH;
                let dh = e % DH;
                let bc_swizzled = bc ^ (((dh >> 3) & 1) * 8);
                unsafe {
                    (*vt_smem.add(dh * BC + bc_swizzled))
                        .write(*v_head.add((kv_base + bc) * DH + dh))
                };
                e += BLOCK_THREADS;
            }
        }
        thread::sync_threads();

        // ── QK^T: S[16,16] = Q_smem·K_smem^T / √Dh ─────────────────────────
        // All 4 warps compute the same S (redundant, but each warp needs its own copy
        // for independent online-softmax state and O accumulation).
        let ksp = addr_of!(K_SMEM) as *const u16;
        let qsp_u16 = addr_of!(Q_SMEM) as *const u16;

        let mut s_j0 = [0.0f32; 4]; // n-tile j=0: KV rows 0-7
        let mut s_j1 = [0.0f32; 4]; // n-tile j=1: KV rows 8-15

        // Precomputed per-lane constants (hoisted out of k-loop by compiler):
        //   krow0 = K row for j=0 sub-matrix (0..7); krow1 = same for j=1 (8..15).
        //   kbase = (lane&8): 0 for lanes 0-7 (matrix 0), 8 for lanes 8-15 (matrix 1).
        let krow0 = lane & 7; // 0..7
        let krow1 = krow0 + 8; // 8..15
        let kbase = lane & 8; // 0 or 8

        let mut kk = 0usize;
        while kk < DH / 16 {
            let kb = kk * 16;
            // A-fragment: ldmatrix.x4 from Q_smem.
            let my_q_row = lane % 16;
            let my_k_off = kb + (lane / 16) * 8;
            let a_addr = unsafe { qsp_u16.add(my_q_row * DH + my_k_off) as u64 };
            let a = unsafe { ldmatrix_a4(a_addr) };

            // B-fragments: 2 × ldmatrix.x2.trans (replaces 4 scalar ld_f16x2).
            let [b0j0, b1j0] = unsafe { ldmatrix_b2(ksp.add(krow0 * DH + kb + kbase) as u64) };
            s_j0 = unsafe { mma_f16(a, [b0j0, b1j0], s_j0) };
            let [b0j1, b1j1] = unsafe { ldmatrix_b2(ksp.add(krow1 * DH + kb + kbase) as u64) };
            s_j1 = unsafe { mma_f16(a, [b0j1, b1j1], s_j1) };
            kk += 1;
        }
        s_j0 = s_j0.map(|x| x * scale);
        s_j1 = s_j1.map(|x| x * scale);

        // ── Online softmax (per-warp; all warps produce identical m/l) ───────
        let lm_g = {
            let a = if s_j0[0] > s_j0[1] { s_j0[0] } else { s_j0[1] };
            let b = if s_j1[0] > s_j1[1] { s_j1[0] } else { s_j1[1] };
            if a > b { a } else { b }
        };
        let lm_g8 = {
            let a = if s_j0[2] > s_j0[3] { s_j0[2] } else { s_j0[3] };
            let b = if s_j1[2] > s_j1[3] { s_j1[2] } else { s_j1[3] };
            if a > b { a } else { b }
        };
        let m_tile_g = unsafe { group_max(lm_g) };
        let m_tile_g8 = unsafe { group_max(lm_g8) };
        let m_new_g = if m_tile_g > m_g { m_tile_g } else { m_g };
        let m_new_g8 = if m_tile_g8 > m_g8 { m_tile_g8 } else { m_g8 };

        let p00 = (s_j0[0] - m_new_g).exp();
        let p01 = (s_j0[1] - m_new_g).exp();
        let p08 = (s_j1[0] - m_new_g).exp();
        let p09 = (s_j1[1] - m_new_g).exp();
        let p80 = (s_j0[2] - m_new_g8).exp();
        let p81 = (s_j0[3] - m_new_g8).exp();
        let p88 = (s_j1[2] - m_new_g8).exp();
        let p89 = (s_j1[3] - m_new_g8).exp();

        let l_tile_g = unsafe { group_sum(p00 + p01 + p08 + p09) };
        let l_tile_g8 = unsafe { group_sum(p80 + p81 + p88 + p89) };

        let rescale_g = (m_g - m_new_g).exp();
        let rescale_g8 = (m_g8 - m_new_g8).exp();

        m_g = m_new_g;
        m_g8 = m_new_g8;
        l_g = l_g * rescale_g + l_tile_g;
        l_g8 = l_g8 * rescale_g8 + l_tile_g8;

        // Rescale O registers (pure register ops — no smem traffic).
        for v in &mut o_g {
            *v *= rescale_g;
        }
        for v in &mut o_g8 {
            *v *= rescale_g8;
        }

        // ── PV: o_g/o_g8 += P · V_T_smem^T ─────────────────────────────────
        // P packed from registers — no P_SMEM. Fragment registers:
        //   a0: row=grp,   k=l2*2..+1 (BC col 0-7)   a2: row=grp,   k=l2*2+8..+9
        //   a1: row=grp+8, k=l2*2..+1                 a3: row=grp+8, k=l2*2+8..+9
        let pa = unsafe {
            [
                pack_f16(p00, p01),
                pack_f16(p80, p81),
                pack_f16(p08, p09),
                pack_f16(p88, p89),
            ]
        };

        // Warp w handles V n-tiles t = [t_base, t_base + VTILES_PER_WARP).
        // o_g[tt*2+r] accumulates O[grp, dh_base+tt*8+l2*2+r]; tt = t - t_base ∈ [0,4).
        // B-fragment from V_T_SMEM [DH, BC] col-major: V_T[dh, bc] = V[bc, dh].
        // Lane l provides column (l&7) of B sub-tile, rows start at (l&8), both swizzled:
        //   vt_row = t*8 + (l&7); vt_col_swizzled = (l&8) ^ (((vt_row >> 3) & 1) * 8)
        let vtsp = addr_of!(V_T_SMEM) as *const u16;
        let vt_lane_koff = lane & 8; // 0 for lanes 0-7, 8 for lanes 8-15
        let mut t = t_base;
        while t < t_base + VTILES_PER_WARP {
            let tt = t - t_base;
            let vt_row = t * 8 + (lane & 7);
            let vt_col_swizzled = vt_lane_koff ^ (((vt_row >> 3) & 1) * 8);
            let vt_addr = unsafe { vtsp.add(vt_row * BC + vt_col_swizzled) as u64 };
            let [b0, b1] = unsafe { ldmatrix_b2(vt_addr) };
            let d = unsafe { mma_f16(pa, [b0, b1], [0.0f32; 4]) };
            o_g[tt * 2] += d[0];
            o_g[tt * 2 + 1] += d[1];
            o_g8[tt * 2] += d[2];
            o_g8[tt * 2 + 1] += d[3];
            t += 1;
        }
        // Barrier: protects V_T_SMEM/K_SMEM from being overwritten by next iteration's load.
        thread::sync_threads();

        kv_tile += 1;
    }

    // ── Normalize O = O / l (pure register ops) ─────────────────────────────
    for v in &mut o_g {
        *v /= l_g;
    }
    for v in &mut o_g8 {
        *v /= l_g8;
    }

    // ── Scatter O to global (each thread writes its 16 f16 cells directly) ───
    // No smem round-trip. Each thread owns dh = dh_base + tt*8 + l2*2 (+1) for tt=0..3.
    let out_base = query_base * DH;
    let mut tt = 0usize;
    while tt < VTILES_PER_WARP {
        let dh0 = dh_base + tt * 8 + l2 * 2;
        let dh1 = dh0 + 1;
        unsafe {
            *o_head.add(out_base + grp * DH + dh0) = cvt_f32_f16(o_g[tt * 2]);
            *o_head.add(out_base + grp * DH + dh1) = cvt_f32_f16(o_g[tt * 2 + 1]);
            *o_head.add(out_base + (grp + 8) * DH + dh0) = cvt_f32_f16(o_g8[tt * 2]);
            *o_head.add(out_base + (grp + 8) * DH + dh1) = cvt_f32_f16(o_g8[tt * 2 + 1]);
        }
        tt += 1;
    }
}
