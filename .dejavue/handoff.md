# Handoff

Updated: 2026-06-22T20:46:00-05:00

## Summary
Flash attention arc. v2 committed at 3e71459 on main.

**v2 results (5070 Ti, Dh=128, f16):**
- 512×512: 0.096 ms, 1.4 TFLOP/s (was 0.5 TFLOP/s in v0)
- 1024×1024: 0.275 ms, 1.9 TFLOP/s
- 2048×2048: 0.755 ms, 2.8 TFLOP/s
- L2-rel ~3.4e-4 (f16 quantization noise; algorithm is correct)

**v2 changes:** 4-warp DH-split (all 4 warps compute same S[16,16], each owns Dh/4=32 slice of O), ldmatrix.x4 for Q A-fragments, register-packed P (no P_SMEM), 128-thread cooperative K/V loads.

**Key constraint:** ldmatrix in PTX .address_size 64 mode (sm_100 target) takes a 64-bit generic address — NOT cvta.to.shared.u32 (which only works with u32 operands in .address_size 32 mode). Pass `ptr as u64` directly.

**Occupancy:** 48 KB smem / 20 KB per block = 2 blocks/SM → 8 active warps/SM (vs 1 in v0). Still only ~25% warp occupancy; headroom remains.

## Next Steps (attention arc)
1) **ldmatrix.x2.trans for K** — replace 4 scalar ld_f16x2 per k-step/n-tile with 1 ldmatrix.x2.trans. Each n-tile j requires separate call (lane ptr formula differs). Saves 4 scalar smem reads per mma.
2) **cp.async pipelining** — overlap K/V global load with QK^T compute using cp.async.cg and cp.async.wait_group. Hides ~100% of global memory latency.
3) **Larger Br** — Br=32 (2 passes of 16 rows per warp) or Br=64 with 8 warps. Need to increase smem or switch O_smem to f16.
4) **Multi-head** — add head-dim stride to Q/K/V/O; outer grid over (head, L/Br).

## GEMV arc (previous, uncommitted)
GEMV edits were NOT committed (v3 Q4_K, Q6_K warp, bench rows). If resuming GEMV: see `.dejavue/state.md` and `.dejavue/decisions.md` — all context is there.

## Boot Instructions
Read `.dejavue/handoff.md`, `.dejavue/state.md`, `.dejavue/decisions.md`, and `.dejavue/timeline.jsonl` before making changes.
