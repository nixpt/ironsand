# Handoff

Updated: 2026-06-22T22:00:00-05:00

## Summary
Flash attention arc. v4 committed at b9db964 on main.

**v4 results (5070 Ti, Dh=128, f16):**
- 512×512: 0.076 ms, 1.8 TFLOP/s
- 1024×1024: 0.201 ms, 2.7 TFLOP/s
- 2048×2048: 0.508 ms, 4.2 TFLOP/s (was 2.8 TFLOP/s in v2 baseline)
- L2-rel ~3.5e-4 (f16 quantization noise; algorithm is correct)

**v4 changes:** Eliminated O_SMEM (8 KB f32 → 0). O lives in per-thread registers
`o_g[8]` / `o_g8[8]`. Rescale, accumulate, and normalize are pure register ops.
Smem: 20 KB → 12 KB → 4 blocks/SM → 16 warps/SM (50% occupancy, was 25%).
Final write: scattered per-thread stores (16 f16 cells/thread) rather than coalesced
smem→global pass.

**v3 changes (bf561f9):** `ldmatrix.x2` (no trans) for K B-fragments (2 calls/k-step)
and V_T B-fragments (1 call/n-tile), replacing scalar `ld_f16x2` loads.
Note: `ldmatrix.x2.trans` was tried and gave 40% L2 error — `trans` expects col-major
source, but K_smem is row-major. Correct instruction is without `.trans`.

**Key constraints:**
- ldmatrix in PTX .address_size 64 mode (sm_100 target) takes 64-bit generic address
  — NOT cvta.to.shared.u32. Pass `ptr as u64` directly.
- `ldmatrix.x2.trans` is WRONG for row-major K_smem. Use plain `ldmatrix.x2`.

## Next Steps (attention arc)
1) **cp.async pipelining** — overlap K/V global load with QK^T compute using
   `cp.async.cg` + `cp.async.wait_group`. L2 for 2048×2048 is ~2 MB (K+V);
   smem double-buffering requires ~24 KB (2×12 KB). May help if not already L2-resident.
2) **Larger Br** — Br=32 with 8 warps (BR=32, NWARPS=8, BLOCK_THREADS=256).
   Would need 3×8 KB = 24 KB smem (still 2 blocks/SM at 48 KB), but 2× more rows/block.
3) **Multi-head** — add head-dim stride to Q/K/V/O; outer grid over (head, L/Br).
4) **Warp-specialized QK^T** — only 2 warps do QK^T, 4 do PV (with shared softmax via smem).

## GEMV arc (previous, uncommitted)
GEMV edits were NOT committed (v3 Q4_K, Q6_K warp, bench rows). If resuming GEMV: see `.dejavue/state.md` and `.dejavue/decisions.md` — all context is there.

## Boot Instructions
Read `.dejavue/handoff.md`, `.dejavue/state.md`, `.dejavue/decisions.md`, and `.dejavue/timeline.jsonl` before making changes.
