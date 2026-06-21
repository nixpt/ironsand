# State

Updated: 2026-06-20T20:42:49-05:00

GEMV series plateau reached. Final state: Q4_K v3 is the production Q4_K (224-269 GB/s, +4-13% over fast). Q6_K warp is the production Q6_K (~380 GB/s, ~21% of 5070 Ti peak). gemv_q4k_v4 (2-way unroll) and gemv_q6k_fast (FMA + unroll) are documented no-wins kept in tree as experiments. Profiling attempted but blocked by RmProfilingAdminOnly=1 (root needed for ncu perf counters); nsys shows kernel timing only, no stall analysis. The compute-side optimization knobs (FMA, unroll, vectorized scale reads, pair-restructure qs reads) are exhausted at the warp-per-row organization. Real next wins likely need: (a) profile-driven optimization (ncu once sudo is available), (b) lane-restructured Q4_K v5 (1 warp per super-block, 8 weights/lane, u32 qs reads, cross-warp reduction), or (c) pivot to next arc (Q5_K, attention, fused kernels, zorro integration).
