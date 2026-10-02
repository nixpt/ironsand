//! Stage-0 spike: prove `rustc_codegen_nvvm`/LLVM19 can emit an int8 **tensor-core**
//! `mma.sync` via inline PTX `asm!` — the go/no-go for a fused Q4_K int8 mmq GEMM.
//!
//! One warp computes a single `m16n8k32.s8` tile:
//!   `C[16×8] (s32) = A[16×32] (s8) · B[8×32]ᵀ (s8)`.
//! ironsand's `cuda_std` exposes no wmma/mma intrinsics, so the only route to int8
//! tensor cores is hand-emitting `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`
//! (exactly what llama.cpp's mma.cuh does). Fragment register↔lane layout is the
//! canonical Ampere+ map (PTX ISA): `groupID = lane/4`, `lane2 = lane%4`.
//!   A: 4 regs (4 s8 each) — a0/a2 row=groupID, a1/a3 row=groupID+8; k = lane2·4 (+16)
//!   B: 2 regs — n=groupID; k = lane2·4 (+16)
//!   C/D: 4 s32 regs — c0,c1 row=groupID col=lane2·2+{0,1}; c2,c3 row=groupID+8

#[cfg(target_os = "cuda")]
use core::arch::asm;
use cuda_std::kernel;
use cuda_std::thread;

#[cfg(target_os = "cuda")]
#[inline(always)]
unsafe fn ld_u32(p: *const u8, off: usize) -> u32 {
    unsafe { (p.add(off) as *const u32).read() }
}
#[cfg(not(target_os = "cuda"))]
#[inline(always)]
unsafe fn ld_u32(_p: *const u8, _off: usize) -> u32 {
    0
}

/// One warp = one `m16n8k32` int8 mma tile. `a` = 16×32 row-major s8 (512 B);
/// `b` = 8×32 row-major s8 (`B[n][k]`, 256 B); `c` = 16×8 row-major s32 out.
/// Launch `<<<1, 32>>>`.
///
/// # Safety
/// `a`/`b` sized exactly as above; `c` points to ≥ 16·8 i32; one warp only.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn mma_int8_tile(a: &[u8], b: &[u8], c: *mut i32) {
    let lane = thread::thread_idx_x() as usize;
    let grp = lane / 4; // groupID 0..7
    let l2 = lane % 4; // threadID-in-group 0..3
    let ap = a.as_ptr();
    let bp = b.as_ptr();

    // A fragment (row-major): a0/a2 = row groupID (k 0-15 / 16-31),
    // a1/a3 = row groupID+8. Each reg packs 4 consecutive-k s8.
    let a0 = unsafe { ld_u32(ap, grp * 32 + l2 * 4) };
    let a1 = unsafe { ld_u32(ap, (grp + 8) * 32 + l2 * 4) };
    let a2 = unsafe { ld_u32(ap, grp * 32 + l2 * 4 + 16) };
    let a3 = unsafe { ld_u32(ap, (grp + 8) * 32 + l2 * 4 + 16) };
    // B fragment (col=n=groupID): b0 = k 0-15, b1 = k 16-31.
    let b0 = unsafe { ld_u32(bp, grp * 32 + l2 * 4) };
    let b1 = unsafe { ld_u32(bp, grp * 32 + l2 * 4 + 16) };

    let mut d0: i32 = 0;
    let mut d1: i32 = 0;
    let mut d2: i32 = 0;
    let mut d3: i32 = 0;

    #[cfg(target_os = "cuda")]
    unsafe {
        // PTX vec braces `{...}` are escaped `{{ ... }}`; `{0}`..`{9}` are operands.
        // Accumulator C and result D share regs (inout) seeded to 0.
        asm!(
            "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{0}, {1}, {2}, {3}}}, {{{4}, {5}, {6}, {7}}}, {{{8}, {9}}}, {{{0}, {1}, {2}, {3}}};",
            inout(reg32) d0,
            inout(reg32) d1,
            inout(reg32) d2,
            inout(reg32) d3,
            in(reg32) a0,
            in(reg32) a1,
            in(reg32) a2,
            in(reg32) a3,
            in(reg32) b0,
            in(reg32) b1,
        );
    }

    // C/D store: 4 s32 regs at (groupID|+8, lane2·2 + {0,1}). c is 16×8 row-major.
    unsafe {
        *c.add(grp * 8 + l2 * 2) = d0;
        *c.add(grp * 8 + l2 * 2 + 1) = d1;
        *c.add((grp + 8) * 8 + l2 * 2) = d2;
        *c.add((grp + 8) * 8 + l2 * 2 + 1) = d3;
    }
}
