//! FlashAttention-2-style prefill v2: 4-warp DH-split + ldmatrix for Q + register-packed P.
//!
//! `O[L, Dh] = softmax(Q[L,Dh] · K[S,Dh]^T / √Dh) · V[S, Dh]`
//!
//! Fixed: Dh=128, Br=16 (query rows/block), Bc=16 (KV columns/step). 4 warps per block.
//! Grid: `(⌈L/16⌉,)` blocks × 128 threads. L and S must be multiples of 16.
//!
//! ## Parallelism strategy
//!
//! **4 warps per block** with identical smem layout (still 20 KB — same as v0):
//!   - All 4 warps compute the same S[16,16] = Q·K^T (redundant but cheap).
//!   - Warp w owns the Dh-slice [w*32, (w+1)*32) of O and 4 PV n-tiles starting at t=w*4.
//!   - No inter-warp communication needed for QK^T/softmax (all produce identical results).
//!   - Race-free O_smem: each warp writes disjoint Dh cells.
//!
//! Occupancy improvement: 48 KB smem / 20 KB per block = 2 blocks per SM → 8 warps per SM
//! vs 1 in v0. Better latency hiding for K/V global loads.
//!
//! ## v1 → v2 carry-overs
//!   - ldmatrix.sync.aligned.x4.m8n8.shared.b16 for Q A-fragment (still present).
//!   - P packed directly from softmax registers into A-fragment u32s (no P_SMEM).
//!   - K/V loaded cooperatively by all 128 threads (stride 128, 16 loads/thread vs 64).
//!
//! ## Fragment layout (Ampere+, grp=lane/4, l2=lane%4, warp_id=tid/32)
//! QK^T A: ldmatrix→[a0..a3] from Q_smem[l%16, kb+(l/16)*8].
//! QK^T B: scalar ld_f16x2 from K_smem[grp*DH+…].
//! PV A: register-packed P values.
//! PV B: scalar ld_f16x2 from V_T_smem[(t*8+grp)*BC+…], t in [warp_id*4, warp_id*4+4).

use core::mem::MaybeUninit;
use core::ptr::{addr_of, addr_of_mut};
use cuda_std::address_space;
use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::GpuFloat;
#[cfg(target_os = "cuda")]
use core::arch::asm;

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

/// Load 2 packed f16 (u32) from raw u16 pointer at offset `idx`.
#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ld_f16x2(p: *const u16, idx: usize) -> u32 {
    unsafe { (p.add(idx) as *const u32).read() }
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn ld_f16x2(_p: *const u16, _idx: usize) -> u32 {
    0
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

// ── Kernel ──────────────────────────────────────────────────────────────────

/// FlashAttention-2 prefill v2: 4-warp DH-split.
/// `O[L,Dh] = softmax(Q·K^T/√Dh)·V`. f16 in/out.
/// Dh=128 fixed; L and S must be multiples of 16. Launch `<<<ceil(L/16), 128>>>`.
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
) {
    #[address_space(shared)]
    static mut Q_SMEM: [MaybeUninit<u16>; BR * DH] = [MaybeUninit::uninit(); BR * DH];
    #[address_space(shared)]
    static mut K_SMEM: [MaybeUninit<u16>; BC * DH] = [MaybeUninit::uninit(); BC * DH];
    #[address_space(shared)]
    static mut V_T_SMEM: [MaybeUninit<u16>; DH * BC] = [MaybeUninit::uninit(); DH * BC];
    #[address_space(shared)]
    static mut O_SMEM: [MaybeUninit<f32>; BR * DH] = [MaybeUninit::uninit(); BR * DH];

    // Raw smem pointers — avoids Rust 2024 ban on &T/&mut T to static mut.
    let q_smem  = addr_of_mut!(Q_SMEM)   as *mut MaybeUninit<u16>;
    let k_smem  = addr_of_mut!(K_SMEM)   as *mut MaybeUninit<u16>;
    let vt_smem = addr_of_mut!(V_T_SMEM)  as *mut MaybeUninit<u16>;
    let o_smem  = addr_of_mut!(O_SMEM)   as *mut MaybeUninit<f32>;

    let tid        = thread::thread_idx_x() as usize;
    let warp_id    = tid / 32;   // 0..3
    let lane       = tid % 32;   // lane within warp (0..31)
    let grp        = lane / 4;   // groupID 0..7
    let l2         = lane % 4;   // threadID-in-group 0..3
    let qi_tile    = thread::block_idx_x() as usize;
    let query_base = qi_tile * BR;
    if query_base >= l_seq {
        return;
    }

    let scale = 1.0f32 / (DH as f32).sqrt();

    // Each warp owns the Dh-slice [warp_id*DH_PER_WARP, (warp_id+1)*DH_PER_WARP).
    let dh_base = warp_id * DH_PER_WARP; // 0, 32, 64, or 96
    // PV n-tiles: warp w handles t in [warp_id*VTILES_PER_WARP, (warp_id+1)*VTILES_PER_WARP).
    let t_base = warp_id * VTILES_PER_WARP; // 0, 4, 8, or 12

    // ── Load Q tile; zero O_SMEM (all BLOCK_THREADS cooperate, stride BLOCK_THREADS) ──
    let mut e = tid;
    while e < BR * DH {
        let row = e / DH;
        let dh  = e % DH;
        unsafe { (*q_smem.add(e)).write(*q.as_ptr().add((query_base + row) * DH + dh)) };
        unsafe { (*o_smem.add(e)).write(0.0f32) };
        e += BLOCK_THREADS;
    }
    thread::sync_threads();

    let mut m_g  = f32::NEG_INFINITY;
    let mut m_g8 = f32::NEG_INFINITY;
    let mut l_g  = 0.0f32;
    let mut l_g8 = 0.0f32;

    // ── KV loop ─────────────────────────────────────────────────────────────
    let mut kv_tile = 0usize;
    while kv_tile < s_seq / BC {
        let kv_base = kv_tile * BC;

        // Load K tile [BC, DH] and V transposed (all 128 threads cooperate).
        let mut e = tid;
        while e < BC * DH {
            let row = e / DH;
            let dh  = e % DH;
            unsafe { (*k_smem.add(e)).write(*k.as_ptr().add((kv_base + row) * DH + dh)) };
            e += BLOCK_THREADS;
        }
        let mut e = tid;
        while e < BC * DH {
            let bc = e / DH;
            let dh = e % DH;
            unsafe { (*vt_smem.add(dh * BC + bc)).write(*v.as_ptr().add((kv_base + bc) * DH + dh)) };
            e += BLOCK_THREADS;
        }
        thread::sync_threads();

        // ── QK^T: S[16,16] = Q_smem·K_smem^T / √Dh ─────────────────────────
        // All 4 warps compute the same S (redundant, but each warp needs its own copy
        // for independent online-softmax state and O accumulation).
        let ksp     = addr_of!(K_SMEM) as *const u16;
        let qsp_u16 = addr_of!(Q_SMEM) as *const u16;

        let mut s_j0 = [0.0f32; 4]; // n-tile j=0: KV rows 0-7
        let mut s_j1 = [0.0f32; 4]; // n-tile j=1: KV rows 8-15

        let mut kk = 0usize;
        while kk < DH / 16 {
            let kb = kk * 16;
            // ldmatrix: lane l provides Q_smem[l%16, kb + (l/16)*8].
            let my_q_row = lane % 16;
            let my_k_off = kb + (lane / 16) * 8;
            let a_addr = unsafe { qsp_u16.add(my_q_row * DH + my_k_off) as u64 };
            let a = unsafe { ldmatrix_a4(a_addr) };

            let b0j0 = unsafe { ld_f16x2(ksp, grp * DH + kb + l2 * 2) };
            let b1j0 = unsafe { ld_f16x2(ksp, grp * DH + kb + l2 * 2 + 8) };
            s_j0 = unsafe { mma_f16(a, [b0j0, b1j0], s_j0) };
            let b0j1 = unsafe { ld_f16x2(ksp, (grp + 8) * DH + kb + l2 * 2) };
            let b1j1 = unsafe { ld_f16x2(ksp, (grp + 8) * DH + kb + l2 * 2 + 8) };
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
        let m_tile_g  = unsafe { group_max(lm_g) };
        let m_tile_g8 = unsafe { group_max(lm_g8) };
        let m_new_g   = if m_tile_g  > m_g  { m_tile_g  } else { m_g };
        let m_new_g8  = if m_tile_g8 > m_g8 { m_tile_g8 } else { m_g8 };

        let p00 = (s_j0[0] - m_new_g).exp();
        let p01 = (s_j0[1] - m_new_g).exp();
        let p08 = (s_j1[0] - m_new_g).exp();
        let p09 = (s_j1[1] - m_new_g).exp();
        let p80 = (s_j0[2] - m_new_g8).exp();
        let p81 = (s_j0[3] - m_new_g8).exp();
        let p88 = (s_j1[2] - m_new_g8).exp();
        let p89 = (s_j1[3] - m_new_g8).exp();

        let l_tile_g  = unsafe { group_sum(p00 + p01 + p08 + p09) };
        let l_tile_g8 = unsafe { group_sum(p80 + p81 + p88 + p89) };

        let rescale_g  = (m_g  - m_new_g).exp();
        let rescale_g8 = (m_g8 - m_new_g8).exp();

        m_g  = m_new_g;
        m_g8 = m_new_g8;
        l_g  = l_g  * rescale_g  + l_tile_g;
        l_g8 = l_g8 * rescale_g8 + l_tile_g8;

        // Rescale O_SMEM: each warp only touches its own Dh-slice [dh_base, dh_base+32).
        // Thread (grp, l2) owns dh = dh_base+l2, dh_base+l2+4, … within that slice.
        let mut dh = dh_base + l2;
        while dh < dh_base + DH_PER_WARP {
            unsafe {
                let ov = (*o_smem.add(grp * DH + dh)).assume_init();
                (*o_smem.add(grp * DH + dh)).write(ov * rescale_g);
                let ov = (*o_smem.add((grp + 8) * DH + dh)).assume_init();
                (*o_smem.add((grp + 8) * DH + dh)).write(ov * rescale_g8);
            }
            dh += 4;
        }

        // ── PV: O_smem += P · V_T_smem^T ────────────────────────────────────
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

        // Warp w handles V_T n-tiles t = [t_base, t_base + VTILES_PER_WARP).
        // O addresses for warp w: O_smem[grp*DH + t*8 + l2*2..] ∈ [dh_base, dh_base+32). ✓
        let vtsp = addr_of!(V_T_SMEM) as *const u16;
        let mut t = t_base;
        while t < t_base + VTILES_PER_WARP {
            let b0 = unsafe { ld_f16x2(vtsp, (t * 8 + grp) * BC + l2 * 2) };
            let b1 = unsafe { ld_f16x2(vtsp, (t * 8 + grp) * BC + l2 * 2 + 8) };
            let d = unsafe { mma_f16(pa, [b0, b1], [0.0f32; 4]) };
            unsafe {
                let ov = (*o_smem.add(grp * DH + t * 8 + l2 * 2)).assume_init();
                (*o_smem.add(grp * DH + t * 8 + l2 * 2)).write(ov + d[0]);
                let ov = (*o_smem.add(grp * DH + t * 8 + l2 * 2 + 1)).assume_init();
                (*o_smem.add(grp * DH + t * 8 + l2 * 2 + 1)).write(ov + d[1]);
                let ov = (*o_smem.add((grp + 8) * DH + t * 8 + l2 * 2)).assume_init();
                (*o_smem.add((grp + 8) * DH + t * 8 + l2 * 2)).write(ov + d[2]);
                let ov = (*o_smem.add((grp + 8) * DH + t * 8 + l2 * 2 + 1)).assume_init();
                (*o_smem.add((grp + 8) * DH + t * 8 + l2 * 2 + 1)).write(ov + d[3]);
            }
            t += 1;
        }
        // Barrier: all warps must finish PV (writes V_T_smem/O_smem) before next K/V load.
        thread::sync_threads();

        kv_tile += 1;
    }

    // ── Normalize O = O / l (each warp handles its Dh slice) ────────────────
    let mut dh = dh_base + l2;
    while dh < dh_base + DH_PER_WARP {
        unsafe {
            let ov = (*o_smem.add(grp * DH + dh)).assume_init();
            (*o_smem.add(grp * DH + dh)).write(ov / l_g);
            let ov = (*o_smem.add((grp + 8) * DH + dh)).assume_init();
            (*o_smem.add((grp + 8) * DH + dh)).write(ov / l_g8);
        }
        dh += 4;
    }
    thread::sync_threads();

    // ── Write f16 to global (all BLOCK_THREADS cooperate, stride BLOCK_THREADS) ──
    let out_base = query_base * DH;
    let mut e = tid;
    while e < BR * DH {
        let val = unsafe { (*o_smem.add(e)).assume_init() };
        unsafe { *o.add(out_base + e) = cvt_f32_f16(val) };
        e += BLOCK_THREADS;
    }
}
