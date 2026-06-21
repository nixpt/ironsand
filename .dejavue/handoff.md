# Handoff

Updated: 2026-06-20T20:42:57-05:00

## Summary
GEMV optimization series done at the warp-per-row level. v3 is production for Q4_K (224-269 GB/s); Q6_K warp is production (~380 GB/s). Plateau: compute-side knobs exhausted, profiling blocked by RmProfilingAdminOnly. Branch: clean, uncommitted (all 8 GEMV-related edits are unstaged).

## Next Steps
1) Decide whether to commit the GEMV series (Q6_K fast, Q4_K v3, Q4_K v4, bench rows, lib.rs exports, dejavue captures) as a single commit, or split per-kernel
2) Optional: drop gemv_q4k_v4 and gemv_q6k_fast from main.rs bench (keep kernels in tree but don't time them) — reduces bench runtime by ~30%
3) Optional: archive v4 and fast to examples/gemv/kernels/archive/ to declutter lib.rs
4) Profile path: when sudo is available, run 'echo 0 | sudo tee /proc/driver/nvidia/params/RmProfilingAdminOnly' and re-run ncu SpeedOfLight + MemoryWorkloadAnalysis on gemv_q4k_v3 and gemv_q6k_warp at 4096x4096
5) Lane-restructured v5 is the next big experiment if profiling doesn't unlock a cheaper win — 1 warp per super-block, 8 weights/lane, u32 qs reads, cross-warp reduction via atomic add to y
6) Pivot candidates: Q5_K kernel (new format), attention kernel (the original next-arc), zorro decode integration (use Q4_K v3 + Q6_K warp as the runtime kernels), or W4A8 (int8 activation × Q4_K weight) which is what zorro actually does in decode

## Boot Instructions
Read `.dejavue/handoff.md`, `.dejavue/state.md`, `.dejavue/decisions.md`, and `.dejavue/timeline.jsonl` before making changes.
