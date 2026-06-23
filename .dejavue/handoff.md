# Handoff

Updated: 2026-06-23T02:00:00-05:00

## Summary
Flash attention arc. v7 committed on main — V_T_SMEM swizzle complete.

**v7 results (5070 Ti, Dh=128, f16):** V_T_SMEM XOR swizzle (phase-8) eliminates 8-way bank conflicts.
- 512×512  H=1:  0.055 ms, 2.4 TFLOP/s
- 1024×1024 H=1: 0.159 ms, 3.4 TFLOP/s
- 2048×2048 H=1: 0.404 ms, 5.3 TFLOP/s  (+4% vs v6 5.1)
- 512×512  H=32: 0.704 ms, 6.1 TFLOP/s
- 1024×1024 H=32: 2.722 ms, 6.3 TFLOP/s
- 2048×2048 H=32: 10.800 ms, 6.4 TFLOP/s  (+10% vs v6 5.8)
- L2-rel ~3.4e-4 (f16 quantization noise; algorithm correct)

**v7 changes:** Phase-8 XOR swizzle on V_T_SMEM: `bc_swizzled = bc ^ (((dh >> 3) & 1) * 8)`.
Applied to both V load scatter and ldmatrix read. Swizzle by multiples of 8 preserves
ldmatrix 16-byte alignment. Eliminates 8-way bank conflict previously accepted.

**V bank-conflict analysis (attempted but reverted):**
- V_T_SMEM[DH, BC] with bc-major vectorized scatter: stride BC*2=32 bytes → all 32 warp
  threads hit bank 0 at each k-step (32-way conflict). H=32 dropped 5.6→5.0 TFLOP/s.
- V_SMEM[BC, DH] (row-major, non-transposed): enables vectorized loads (same pattern as K),
  but ldmatrix formula is NOT a simple transpose of the V_T formula. The B-fragment from
  V_T[dh, bc] (columns of V) is fundamentally different from V[bc, dh] (rows of V).
  Several ldmatrix address formulas attempted — all either misaligned or produced wrong results.
  Root cause: ldmatrix.x2 in the PV MMA expects B[K, N] where consecutive bytes in smem
  form a K-column (V_T layout); V_SMEM rows store the N-dimension instead.
- DECISION: Keep V_T_SMEM scalar scatter. 8-way smem bank conflict accepted.

**v5 changes (afe09c7):** Multi-head support with 2D grid (x=query tiles, y=heads).
H=32: ~5.6 TFLOP/s (was 4.1 TFLOP/s for H=1 in v4 due to L2 cache benefit).

**Key constraints:**
- ldmatrix in PTX .address_size 64 mode (sm_100 target) takes 64-bit generic address
  — NOT cvta.to.shared.u32. Pass `ptr as u64` directly.
- `ldmatrix.x2.trans` is WRONG for row-major K_smem. Use plain `ldmatrix.x2`.
- V_T_SMEM[DH, BC]: lane l provides V_T[(t*8 + l&7)*BC + (l&8)] = V[l&8..l&8+7, t*8+l&7].
  This gives a COLUMN of V to each lane (varying bc, fixed dh) — exactly what ldmatrix.x2
  needs for the B-fragment. V_SMEM[BC, DH] rows don't map to this fragment layout.
- ld_global_v4 / st_shared_v4: require 16-byte aligned addresses. Q/K tiles have this
  because BR=BC=16, DH=128 → row stride = 256 bytes, all 8-u16 chunks are 16-byte aligned.

## v7: Final stable release (6.4 TFLOP/s H=32)

## Attempted next steps

**Br=32 (8 warps)** — TESTED, CORRECTNESS BROKEN (L2-rel~0.95). Root cause: All 8 warps
independently compute & apply softmax over the same attention scores. Since each warp's
m_g/l_g diverges (shfl_xor is per-warp), P values differ per-warp, producing inconsistent
output. Simple fix would require cross-warp reduction on softmax state (not available).

Fix requires **warp-specialized QK^T**: 2 warps compute S, apply softmax to smem, then all
warps read P and compute PV. Avoids softmax divergence but adds complexity (barriers,
smem coordination, async across warp groups). Estimated ~20-30% harder to implement correctly.

**cp.async pipelining** — TESTED, NOT BENEFICIAL. K/V loads are not the bottleneck.
FlashAttention's tight load→compute pattern already has good smem locality. Adding async K
+ early V load adds work per iteration without overlap. Net: 20% slower.

## Known limitations
- Softmax computed redundantly by all warps (OK for Br≤16, breaks at Br≥32)
- Bank conflicts in V_T_SMEM reduced to 8-way via swizzle (could go lower with full banked layout)
- No prefetching/pipelining (loads are synchronous but not a bottleneck for current sizes)

## GEMV arc (previous, uncommitted)
GEMV edits were NOT committed (v3 Q4_K, Q6_K warp, bench rows). If resuming GEMV: see `.dejavue/state.md` and `.dejavue/decisions.md` — all context is there.

## Boot Instructions
Read `.dejavue/handoff.md`, `.dejavue/state.md`, `.dejavue/decisions.md`, and `.dejavue/timeline.jsonl` before making changes.
