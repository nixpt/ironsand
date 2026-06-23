# Handoff

Updated: 2026-06-22T23:30:00-05:00

## Summary
Flash attention arc. v6 committed on main.

**v6 results (5070 Ti, Dh=128, f16):**
- 512×512  H=1:  0.055 ms, 2.4 TFLOP/s
- 1024×1024 H=1: 0.163 ms, 3.3 TFLOP/s
- 2048×2048 H=1: 0.422 ms, 5.1 TFLOP/s  (+24% vs v5 4.1)
- 512×512  H=32: 0.765 ms, 5.6 TFLOP/s
- 1024×1024 H=32: 2.965 ms, 5.8 TFLOP/s
- 2048×2048 H=32: 11.783 ms, 5.8 TFLOP/s
- L2-rel ~3.5e-4 (f16 quantization noise; algorithm correct)

**v6 changes:** Vectorized Q and K tile loads (Q_SMEM, K_SMEM) using 128-bit global
loads (`ld.global.v4.b32`) and 128-bit smem stores (`st.shared.v4.b32`). Each thread
does 2 vectorized loads/stores per tile (covering 8 u16 per load), replacing 16-iteration
scalar while-loops. V load unchanged (V_T_SMEM scalar scatter; see V bank-conflict note).

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

## Next Steps (attention arc)
1) **cp.async pipelining** — overlap K/V global load with QK^T compute using
   `cp.async.cg` + `cp.async.wait_group`. L2 for 2048×2048 is ~2 MB (K+V);
   smem double-buffering requires ~24 KB (2×12 KB). May help if not already L2-resident.
2) **Larger Br** — Br=32 with 8 warps (BR=32, NWARPS=8, BLOCK_THREADS=256).
   Would need 3×8 KB = 24 KB smem (still 2 blocks/SM at 48 KB), but 2× more rows/block.
3) **V smem swizzle** — swizzle V_T_SMEM addressing to eliminate 8-way bank conflicts.
   XOR swizzle: `smem_offset ^= (bc >> 3) << 3` on both store and ldmatrix address.
4) **Warp-specialized QK^T** — only 2 warps do QK^T, 4 do PV (with shared softmax via smem).

## GEMV arc (previous, uncommitted)
GEMV edits were NOT committed (v3 Q4_K, Q6_K warp, bench rows). If resuming GEMV: see `.dejavue/state.md` and `.dejavue/decisions.md` — all context is there.

## Boot Instructions
Read `.dejavue/handoff.md`, `.dejavue/state.md`, `.dejavue/decisions.md`, and `.dejavue/timeline.jsonl` before making changes.
