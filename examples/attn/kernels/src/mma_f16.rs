//! Stage-0 spike: prove `rustc_codegen_nvvm`/LLVM19 can emit a **f16** tensor-core
//! `mma.sync` via inline PTX `asm!` — the prerequisite for FlashAttention-2 prefill.
//!
//! One warp computes a single `m16n8k16.f16` tile:
//!   `D[16×8] (f32) = A[16×16] (f16) · B[16×8] (f16, col-major)`.
//!
//! Fragment register layout (canonical Ampere+, PTX ISA §9.7.13.4):
//!   `groupID = lane/4`, `lane2 = lane%4`.
//!   A [m=16, k=16, row-major]: 4 regs of 2×f16
//!     a0: row=groupID,   k=lane2*2..+1   a2: row=groupID,   k=lane2*2+8..+9
//!     a1: row=groupID+8, k=lane2*2..+1   a3: row=groupID+8, k=lane2*2+8..+9
//!   B [k=16, n=8, col-major]: 2 regs of 2×f16
//!     b0: col=groupID, k=lane2*2..+1     b1: col=groupID, k=lane2*2+8..+9
//!   C/D [m=16, n=8, f32]: 4 regs
//!     d0: (groupID, lane2*2)   d1: (groupID, lane2*2+1)
//!     d2: (groupID+8, lane2*2) d3: (groupID+8, lane2*2+1)

#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::kernel;
use cuda_std::thread;

/// Load 2 packed f16 values from a u16 slice starting at `idx` (u16 index).
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

/// `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` — one f16 tensor-core tile.
/// A = 4 packed-f16 regs, B = 2 packed-f16 regs; C/D = 4 f32 regs.
#[cfg(target_os = "cuda")]
#[inline(always)]
pub unsafe fn mma_f16(a: [u32; 4], b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    let (mut d0, mut d1, mut d2, mut d3) = (c[0], c[1], c[2], c[3]);
    unsafe {
        asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{0},{1},{2},{3}}}, {{{4},{5},{6},{7}}}, {{{8},{9}}}, {{{0},{1},{2},{3}}};",
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
pub unsafe fn mma_f16(_a: [u32; 4], _b: [u32; 2], c: [f32; 4]) -> [f32; 4] {
    c
}

/// One warp = one m16n8k16 f16→f32 mma tile. A = [M=16, K=16] f16 row-major (u16 bits);
/// B = [K=16, N=8] f16 **col-major** (B_col[n*K + k] = B[k,n]); C = [M=16, N=8] f32.
/// Launch `<<<1, 32>>>`.
///
/// # Safety
/// `a` = 16*16 u16; `b` = 16*8 u16 in col-major; `c` = 16*8 f32 out.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn mma_f16_tile(a: &[u16], b: &[u16], c: *mut f32) {
    const M: usize = 16;
    const N: usize = 8;
    const K: usize = 16;

    let lane = thread::thread_idx_x() as usize;
    let grp = lane / 4; // groupID 0..7
    let l2 = lane % 4; // threadID-in-group 0..3
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    // A [M=16, K=16] row-major: element (row, col) at row*K + col.
    let a0 = unsafe { ld_f16x2(ap, grp * K + l2 * 2) };
    let a1 = unsafe { ld_f16x2(ap, (grp + 8) * K + l2 * 2) };
    let a2 = unsafe { ld_f16x2(ap, grp * K + l2 * 2 + 8) };
    let a3 = unsafe { ld_f16x2(ap, (grp + 8) * K + l2 * 2 + 8) };

    // B [K=16, N=8] col-major: element (k, n) at n*K + k.
    // b0/b1 for col=grp: B_col[grp*K + l2*2] and [grp*K + l2*2+8].
    let b0 = unsafe { ld_f16x2(bp, grp * K + l2 * 2) };
    let b1 = unsafe { ld_f16x2(bp, grp * K + l2 * 2 + 8) };

    let d = unsafe { mma_f16([a0, a1, a2, a3], [b0, b1], [0.0f32; 4]) };

    // D [M=16, N=8] row-major: d0(grp, l2*2), d1(grp, l2*2+1), d2(grp+8, l2*2), d3.
    unsafe {
        *c.add(grp * N + l2 * 2) = d[0];
        *c.add(grp * N + l2 * 2 + 1) = d[1];
        *c.add((grp + 8) * N + l2 * 2) = d[2];
        *c.add((grp + 8) * N + l2 * 2 + 1) = d[3];
    }
    let _ = (M, N);
}
