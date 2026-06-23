# Hybrid Kernel Design Session — Memory Index

**Session Date**: 2026-06-23  
**Status**: Design complete, 4-week shipping plan ready

---

## Core Architecture (Read These First)

- [Complete Kernel Vision](complete_kernel_vision.md) — 5-stage evolution from monolithic failure → speculative execution
- [Haiku-San Capacity Analysis](haiku_san_capacity_analysis.md) — Why 64 kernels/token is optimal (CPU overhead, GPU event limits, occupancy)
- [Role-Based Kernels Design](role_based_kernels_design.md) — Phase-aware specialization (prefill vs decode)

---

## Improvements & Extensions

- [Hybrid Kernel Extensions](hybrid_kernel_extensions.md) — 5 major improvements (pipeline, streaming, speculation, fusion, dynamic batching)
- [Improvements Summary](improvements_summary.md) — Quick reference table + stacking effects
- [Learned Kernel Optimization](learned_kernel_optimization.md) — Neural meta-orchestration (4 tiny models, <4KB total)

---

## Implementation Details

- [Stream Kernel Design](stream_kernel_design.md) — Queue-based persistent dispatch (opcodes 0-3, 10-11)
- [Haiku-San Kernel Design](haiku_san_kernel_design.md) — CPU/GPU hybrid orchestration (64 tasks, dependency tracking)

---

## Shipping (THIS MONTH)

- **[4-Week Implementation Plan](4week_implementation_plan.md)** ← START HERE
  - Week 1: Implement 3 decode roles
  - Week 2: Integrate into Haiku-San
  - Week 3: Wire into zorro decode loop
  - Week 4: Measure & ship
  - Target: 20% decode speedup by July 21

---

## Session Summary

- [Session Final Summary](session_final_summary.md) — Complete overview of design work

---

## Key Files (Code)

### Kernel Implementation
- `/workspace/projects/ironsand/examples/attn/kernels/src/stream_kernel.rs` — Persistent GPU dispatch (4-op kernel)
- `/workspace/projects/ironsand/examples/attn/kernels/src/lib.rs` — Kernel exports

### Host Orchestration
- `/workspace/projects/ironsand/examples/attn/src/haiku_san.rs` — CPU/GPU hybrid (64-task management)
- `/workspace/projects/ironsand/examples/attn/src/main.rs` — Test spikes (4 orchestration scenarios)

---

## Quick Navigation

**Ask**: "What do I work on first?"  
**Answer**: Start with [4-Week Implementation Plan](4week_implementation_plan.md), Week 1, Day 1-3 (RoleRMSNormSingle kernel)

**Ask**: "Why this architecture?"  
**Answer**: Read [Complete Kernel Vision](complete_kernel_vision.md) (5 stages, why each matters)

**Ask**: "Can I do this faster / in a different order?"  
**Answer**: Read [4-Week Implementation Plan](4week_implementation_plan.md), section "Fallback: If days fall behind"

**Ask**: "What about prefill / speculation / learned models?"  
**Answer**: Deferred to Phase 2+. Shipping decode roles first (2 weeks) proves the architecture works.

---

## Commit Checklist (4 Weeks)

```
Week 1:
  ✓ RoleRMSNormSingle kernel
  ✓ RoleGEMVDecodeSingle kernel
  ✓ RoleFlashAttnSingle kernel

Week 2:
  ✓ Phase-aware dispatch in Haiku-San
  ✓ KV cache streaming
  ✓ Role correctness tests

Week 3:
  ✓ Integration into zorro decode loop
  ✓ Regression testing (Paris oracle)
  ✓ Integration guide docs

Week 4:
  ✓ Latency benchmarking
  ✓ Configuration docs
  ✓ Release tag (v1.0-decode-roles)
```

Expected final gain: **15-20% decode speedup**

---

## Success Criteria

- ✅ All 3 decode roles working (latency targets met)
- ✅ Integrated into zorro (no regressions)
- ✅ Measured 15-20% speedup
- ✅ Documented for handoff

Ship when all four are true.
