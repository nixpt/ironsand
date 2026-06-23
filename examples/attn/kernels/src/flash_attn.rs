//! FlashAttention-2-style prefill attention kernel using f16 mma.sync.
//!
//! `O[L, Dh] = softmax(Q[L,Dh] · K[S,Dh]^T / √Dh) · V[S, Dh]`
//!
//! Fixed: Dh=128, Br=16 (query rows/block), Bc=16 (KV columns/step). One warp per block.
//! Grid: `(⌈L/16⌉,)` blocks. All tensors f16 (u16 bits). L and S must be multiples of 16.
//!
//! ## Algorithm (per block, per KV step)
//!
//! 1. **QK^T**: S[16,16] = Q_tile[16,128] · K_tile^T[128,16] / √128 via mma.f16.
//! 2. **Online softmax**: rowmax via group-local shfl (XOR 1,2); P = exp(S−m); rescale O.
//! 3. **PV**: O_smem[16,128] += P[16,16] · V_T_smem^T via mma.f16.
//!
//! ## Fragment → smem address map (Ampere+ canonical, grp=lane/4, l2=lane%4)
//! QK^T A from Q_smem[grp*DH + kbase + l2*2] (±8 for a2/a3).
//! QK^T B from K_smem[grp*DH + kbase + l2*2] (grp+8 for tile j=1).
//! PV   A from P_smem[grp*BC + l2*2] (±8 for a2/a3).
//! PV   B from V_T_smem[(t*8+grp)*BC + l2*2] (±8 for b1).

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

/// Max-reduce across 4-thread groups (masks 1 and 2 only).
#[inline(always)]
unsafe fn group_max(v: f32) -> f32 {
    let v = {
        let w = unsafe { shfl_xor_f32(v, 2) };
        if w > v { w } else { v }
    };
    let w = unsafe { shfl_xor_f32(v, 1) };
    if w > v { w } else { v }
}

/// Sum-reduce across 4-thread groups.
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

// ── Kernel ──────────────────────────────────────────────────────────────────

/// FlashAttention-2 prefill: `O[L,Dh] = softmax(Q·K^T/√Dh)·V`. f16 in/out.
/// Dh fixed at 128; L and S must be multiples of 16. One warp per block.
/// Launch: `<<<ceil(L/16), 32>>>`.
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
    static mut P_SMEM: [MaybeUninit<u16>; BR * BC] = [MaybeUninit::uninit(); BR * BC];
    #[address_space(shared)]
    static mut O_SMEM: [MaybeUninit<f32>; BR * DH] = [MaybeUninit::uninit(); BR * DH];

    // Raw pointers to smem — avoids creating references to static mut (Rust 2024).
    let q_smem  = unsafe { addr_of_mut!(Q_SMEM)   as *mut MaybeUninit<u16> };
    let k_smem  = unsafe { addr_of_mut!(K_SMEM)   as *mut MaybeUninit<u16> };
    let vt_smem = unsafe { addr_of_mut!(V_T_SMEM)  as *mut MaybeUninit<u16> };
    let p_smem  = unsafe { addr_of_mut!(P_SMEM)    as *mut MaybeUninit<u16> };
    let o_smem  = unsafe { addr_of_mut!(O_SMEM)    as *mut MaybeUninit<f32> };

    let qi_tile   = thread::block_idx_x() as usize;
    let lane      = thread::thread_idx_x() as usize;
    let grp       = lane / 4;
    let l2        = lane % 4;
    let query_base = qi_tile * BR;
    if query_base >= l_seq {
        return;
    }

    let scale = 1.0f32 / (DH as f32).sqrt();

    // ── Load Q tile; zero O_SMEM ─────────────────────────────────────────────
    let mut e = lane;
    while e < BR * DH {
        let row = e / DH;
        let dh  = e % DH;
        let src = unsafe { *q.as_ptr().add((query_base + row) * DH + dh) };
        unsafe { (*q_smem.add(e)).write(src) };
        unsafe { (*o_smem.add(e)).write(0.0f32) };
        e += 32;
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

        // Load K tile [BC, DH] row-major:
        let mut e = lane;
        while e < BC * DH {
            let row = e / DH;
            let dh  = e % DH;
            let src = unsafe { *k.as_ptr().add((kv_base + row) * DH + dh) };
            unsafe { (*k_smem.add(e)).write(src) };
            e += 32;
        }

        // Load V transposed: V[bc, dh] → V_T_SMEM[dh, bc] = V_T_SMEM[dh*BC + bc].
        let mut e = lane;
        while e < BC * DH {
            let bc = e / DH;
            let dh = e % DH;
            let src = unsafe { *v.as_ptr().add((kv_base + bc) * DH + dh) };
            unsafe { (*vt_smem.add(dh * BC + bc)).write(src) };
            e += 32;
        }
        thread::sync_threads();

        // ── QK^T ────────────────────────────────────────────────────────────
        // Two n=8 tiles (j=0: KV rows 0-7, j=1: KV rows 8-15) × 8 k=16 steps.
        let qsp = unsafe { addr_of!(Q_SMEM) as *const u16 };
        let ksp = unsafe { addr_of!(K_SMEM) as *const u16 };

        let mut s_j0 = [0.0f32; 4];
        let mut s_j1 = [0.0f32; 4];

        let mut kk = 0usize;
        while kk < DH / 16 {
            let kb = kk * 16;
            let a0 = unsafe { ld_f16x2(qsp, grp * DH + kb + l2 * 2) };
            let a1 = unsafe { ld_f16x2(qsp, (grp + 8) * DH + kb + l2 * 2) };
            let a2 = unsafe { ld_f16x2(qsp, grp * DH + kb + l2 * 2 + 8) };
            let a3 = unsafe { ld_f16x2(qsp, (grp + 8) * DH + kb + l2 * 2 + 8) };
            let a = [a0, a1, a2, a3];
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

        // ── Online softmax ───────────────────────────────────────────────────
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
        let m_new_g  = if m_tile_g  > m_g  { m_tile_g  } else { m_g };
        let m_new_g8 = if m_tile_g8 > m_g8 { m_tile_g8 } else { m_g8 };

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

        // Rescale O_SMEM: each of 4 threads handles dh=l2,l2+4,... for rows grp, grp+8.
        let mut dh = l2;
        while dh < DH {
            unsafe {
                let v = (*o_smem.add(grp * DH + dh)).assume_init();
                (*o_smem.add(grp * DH + dh)).write(v * rescale_g);
                let v = (*o_smem.add((grp + 8) * DH + dh)).assume_init();
                (*o_smem.add((grp + 8) * DH + dh)).write(v * rescale_g8);
            }
            dh += 4;
        }

        // Write P to P_SMEM (f32→f16). Thread (grp,l2) owns its 4 columns per row.
        unsafe {
            let pp = p_smem as *mut u16;
            *pp.add(grp * BC + l2 * 2)         = cvt_f32_f16(p00);
            *pp.add(grp * BC + l2 * 2 + 1)     = cvt_f32_f16(p01);
            *pp.add(grp * BC + l2 * 2 + 8)     = cvt_f32_f16(p08);
            *pp.add(grp * BC + l2 * 2 + 9)     = cvt_f32_f16(p09);
            *pp.add((grp + 8) * BC + l2 * 2)   = cvt_f32_f16(p80);
            *pp.add((grp + 8) * BC + l2 * 2 + 1) = cvt_f32_f16(p81);
            *pp.add((grp + 8) * BC + l2 * 2 + 8) = cvt_f32_f16(p88);
            *pp.add((grp + 8) * BC + l2 * 2 + 9) = cvt_f32_f16(p89);
        }
        thread::sync_threads();

        // ── PV: O_smem += P · V (via V_T_smem) ──────────────────────────────
        // A from P_SMEM [BR, BC] (same for all 16 n-tiles of Dh).
        let ppsp = unsafe { addr_of!(P_SMEM) as *const u16 };
        let vtsp = unsafe { addr_of!(V_T_SMEM) as *const u16 };

        let a0 = unsafe { ld_f16x2(ppsp, grp * BC + l2 * 2) };
        let a1 = unsafe { ld_f16x2(ppsp, (grp + 8) * BC + l2 * 2) };
        let a2 = unsafe { ld_f16x2(ppsp, grp * BC + l2 * 2 + 8) };
        let a3 = unsafe { ld_f16x2(ppsp, (grp + 8) * BC + l2 * 2 + 8) };
        let pa = [a0, a1, a2, a3];

        let mut t = 0usize;
        while t < DH / 8 {
            // B from V_T_SMEM [DH, BC]: col = grp (= head-dim t*8+grp for this tile).
            let b0 = unsafe { ld_f16x2(vtsp, (t * 8 + grp) * BC + l2 * 2) };
            let b1 = unsafe { ld_f16x2(vtsp, (t * 8 + grp) * BC + l2 * 2 + 8) };
            let d = unsafe { mma_f16(pa, [b0, b1], [0.0f32; 4]) };
            // Accumulate into O_SMEM (unique addresses per thread, no atomics needed).
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
        thread::sync_threads();

        kv_tile += 1;
    }

    // ── Normalize O = O / l, write f16 to global ─────────────────────────────
    let mut dh = l2;
    while dh < DH {
        unsafe {
            let v = (*o_smem.add(grp * DH + dh)).assume_init();
            (*o_smem.add(grp * DH + dh)).write(v / l_g);
            let v = (*o_smem.add((grp + 8) * DH + dh)).assume_init();
            (*o_smem.add((grp + 8) * DH + dh)).write(v / l_g8);
        }
        dh += 4;
    }
    thread::sync_threads();

    let out_base = query_base * DH;
    let mut e = lane;
    while e < BR * DH {
        let val = unsafe { (*o_smem.add(e)).assume_init() };
        unsafe { *o.add(out_base + e) = cvt_f32_f16(val) };
        e += 32;
    }
}
