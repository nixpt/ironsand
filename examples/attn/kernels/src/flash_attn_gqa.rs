//! FlashAttention-2 with Grouped-Query Attention (GQA).
//!
//! Standard attention: H query heads, H KV heads.
//! GQA: H query heads, G KV heads (where G < H, typically G = H/4 or H/8).
//! Multiple query heads share the same KV head group.
//!
//! Layout: Q[H, L, Dh], K[G, S, Dh], V[G, S, Dh]
//! Mapping: query_head h → kv_head = h / (H / G)

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
const DH_PER_WARP: usize = DH / NWARPS;
const VTILES_PER_WARP: usize = (DH / 8) / NWARPS;

// ── Inline-asm helpers (same as flash_attn.rs) ──

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

#[inline(always)]
unsafe fn group_max(v: f32) -> f32 {
    let v = {
        let w = unsafe { shfl_xor_f32(v, 2) };
        if w > v { w } else { v }
    };
    let w = unsafe { shfl_xor_f32(v, 1) };
    if w > v { w } else { v }
}

#[inline(always)]
unsafe fn group_sum(mut v: f32) -> f32 {
    v += unsafe { shfl_xor_f32(v, 2) };
    v += unsafe { shfl_xor_f32(v, 1) };
    v
}

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

/// FlashAttention-2 GQA v1: Grouped-Query Attention.
/// Q[H, L, Dh], K[G, S, Dh], V[G, S, Dh] where G <= H.
/// Multiple query heads share KV heads: query_head h → kv_head = h / (H / G).
///
/// Grid: (query_tiles, query_heads) — more blocks to cover all H query heads,
/// but each block's KV loading is reduced by factor H/G.
///
/// # Safety
/// `L % 16 == 0`, `S % 16 == 0`, `Dh == 128`. `num_query_heads >= num_kv_heads > 0`.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn flash_attn_gqa(
    q: &[u16],
    k: &[u16],
    v: &[u16],
    o: *mut u16,
    l_seq: usize,
    s_seq: usize,
    num_query_heads: usize,
    num_kv_heads: usize,
    // Strides: for Q, stride is l_seq * Dh per head;
    // for K/V, stride is s_seq * Dh per kv-head.
) {
    #[address_space(shared)]
    static mut Q_SMEM: [MaybeUninit<u16>; BR * DH] = [MaybeUninit::uninit(); BR * DH];
    #[address_space(shared)]
    static mut K_SMEM: [MaybeUninit<u16>; BC * DH] = [MaybeUninit::uninit(); BC * DH];
    #[address_space(shared)]
    static mut V_T_SMEM: [MaybeUninit<u16>; DH * BC] = [MaybeUninit::uninit(); DH * BC];

    let q_smem  = addr_of_mut!(Q_SMEM)   as *mut MaybeUninit<u16>;
    let k_smem  = addr_of_mut!(K_SMEM)   as *mut MaybeUninit<u16>;
    let vt_smem = addr_of_mut!(V_T_SMEM)  as *mut MaybeUninit<u16>;

    let tid        = thread::thread_idx_x() as usize;
    let warp_id    = tid / 32;
    let lane       = tid % 32;
    let grp        = lane / 4;
    let l2         = lane % 4;
    let query_head = thread::block_idx_y() as usize;
    let qi_tile    = thread::block_idx_x() as usize;
    let query_base = qi_tile * BR;

    if query_base >= l_seq || query_head >= num_query_heads {
        return;
    }

    // Map query_head to kv_head (GQA grouping).
    let kv_head = query_head * num_kv_heads / num_query_heads;

    let q_head_stride = l_seq * DH;
    let kv_head_stride = s_seq * DH;

    let q_head   = unsafe { q.as_ptr().add(query_head * q_head_stride) };
    let k_head   = unsafe { k.as_ptr().add(kv_head * kv_head_stride) };
    let v_head   = unsafe { v.as_ptr().add(kv_head * kv_head_stride) };
    let o_head   = unsafe { o.add(query_head * q_head_stride) };

    let scale = 1.0f32 / (DH as f32).sqrt();

    let dh_base = warp_id * DH_PER_WARP;
    let t_base  = warp_id * VTILES_PER_WARP;

    // ── Load Q tile ──────────────────────────────────────────────────────────
    {
        let row_a = tid >> 4;
        let col_a = (tid & 15) << 3;
        let v4a = unsafe { ld_global_v4(q_head.add((query_base + row_a) * DH + col_a) as u64) };
        unsafe { st_shared_v4(q_smem.add(row_a * DH + col_a) as u64, v4a) };

        let e_b   = tid + BLOCK_THREADS;
        let row_b = e_b >> 4;
        let col_b = (e_b & 15) << 3;
        let v4b = unsafe { ld_global_v4(q_head.add((query_base + row_b) * DH + col_b) as u64) };
        unsafe { st_shared_v4(q_smem.add(row_b * DH + col_b) as u64, v4b) };
    }
    thread::sync_threads();

    let mut o_g:  [f32; 8] = [0.0; 8];
    let mut o_g8: [f32; 8] = [0.0; 8];

    let mut m_g  = f32::NEG_INFINITY;
    let mut m_g8 = f32::NEG_INFINITY;
    let mut l_g  = 0.0f32;
    let mut l_g8 = 0.0f32;

    // ── KV loop (same as standard attention) ─────────────────────────────────
    let mut kv_tile = 0usize;
    while kv_tile < s_seq / BC {
        let kv_base = kv_tile * BC;

        // Load K (from mapped kv_head)
        {
            let row_a = tid >> 4;
            let col_a = (tid & 15) << 3;
            let v4a = unsafe { ld_global_v4(k_head.add((kv_base + row_a) * DH + col_a) as u64) };
            unsafe { st_shared_v4(k_smem.add(row_a * DH + col_a) as u64, v4a) };

            let e_b   = tid + BLOCK_THREADS;
            let row_b = e_b >> 4;
            let col_b = (e_b & 15) << 3;
            let v4b = unsafe { ld_global_v4(k_head.add((kv_base + row_b) * DH + col_b) as u64) };
            unsafe { st_shared_v4(k_smem.add(row_b * DH + col_b) as u64, v4b) };
        }

        // Load V^T (from mapped kv_head)
        {
            let mut e = tid;
            while e < BC * DH {
                let bc = e / DH;
                let dh = e % DH;
                let bc_swizzled = bc ^ (((dh >> 3) & 1) * 8);
                unsafe { (*vt_smem.add(dh * BC + bc_swizzled)).write(*v_head.add((kv_base + bc) * DH + dh)) };
                e += BLOCK_THREADS;
            }
        }
        thread::sync_threads();

        // ── QK^T ────────────────────────────────────────────────────────────
        let ksp     = addr_of!(K_SMEM) as *const u16;
        let qsp_u16 = addr_of!(Q_SMEM) as *const u16;

        let mut s_j0 = [0.0f32; 4];
        let mut s_j1 = [0.0f32; 4];

        let krow0 = lane & 7;
        let krow1 = krow0 + 8;
        let kbase = lane & 8;

        let mut kk = 0usize;
        while kk < DH / 16 {
            let kb = kk * 16;
            let my_q_row = lane % 16;
            let my_k_off = kb + (lane / 16) * 8;
            let a_addr = unsafe { qsp_u16.add(my_q_row * DH + my_k_off) as u64 };
            let a = unsafe { ldmatrix_a4(a_addr) };

            let [b0j0, b1j0] = unsafe { ldmatrix_b2(ksp.add(krow0 * DH + kb + kbase) as u64) };
            s_j0 = unsafe { mma_f16(a, [b0j0, b1j0], s_j0) };
            let [b0j1, b1j1] = unsafe { ldmatrix_b2(ksp.add(krow1 * DH + kb + kbase) as u64) };
            s_j1 = unsafe { mma_f16(a, [b0j1, b1j1], s_j1) };
            kk += 1;
        }
        s_j0 = s_j0.map(|x| x * scale);
        s_j1 = s_j1.map(|x| x * scale);

        // ── Online softmax ──────────────────────────────────────────────────
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

        for v in &mut o_g  { *v *= rescale_g; }
        for v in &mut o_g8 { *v *= rescale_g8; }

        // ── PV ──────────────────────────────────────────────────────────────
        let pa = unsafe {
            [
                pack_f16(p00, p01),
                pack_f16(p80, p81),
                pack_f16(p08, p09),
                pack_f16(p88, p89),
            ]
        };

        let vtsp = addr_of!(V_T_SMEM) as *const u16;
        let vt_lane_koff = lane & 8;
        let mut t = t_base;
        while t < t_base + VTILES_PER_WARP {
            let tt = t - t_base;
            let vt_row = t * 8 + (lane & 7);
            let vt_col_swizzled = vt_lane_koff ^ (((vt_row >> 3) & 1) * 8);
            let vt_addr = unsafe { vtsp.add(vt_row * BC + vt_col_swizzled) as u64 };
            let [b0, b1] = unsafe { ldmatrix_b2(vt_addr) };
            let d = unsafe { mma_f16(pa, [b0, b1], [0.0f32; 4]) };
            o_g[tt * 2]      += d[0];
            o_g[tt * 2 + 1]  += d[1];
            o_g8[tt * 2]     += d[2];
            o_g8[tt * 2 + 1] += d[3];
            t += 1;
        }

        thread::sync_threads();
        kv_tile += 1;
    }

    // ── Normalize O ──────────────────────────────────────────────────────────
    for v in &mut o_g  { *v /= l_g; }
    for v in &mut o_g8 { *v /= l_g8; }

    // ── Scatter O ────────────────────────────────────────────────────────────
    let out_base = query_base * DH;
    let mut tt = 0usize;
    while tt < VTILES_PER_WARP {
        let dh0 = dh_base + tt * 8 + l2 * 2;
        let dh1 = dh0 + 1;
        unsafe {
            *o_head.add(out_base + grp * DH + dh0)       = cvt_f32_f16(o_g[tt * 2]);
            *o_head.add(out_base + grp * DH + dh1)       = cvt_f32_f16(o_g[tt * 2 + 1]);
            *o_head.add(out_base + (grp + 8) * DH + dh0) = cvt_f32_f16(o_g8[tt * 2]);
            *o_head.add(out_base + (grp + 8) * DH + dh1) = cvt_f32_f16(o_g8[tt * 2 + 1]);
        }
        tt += 1;
    }
}
