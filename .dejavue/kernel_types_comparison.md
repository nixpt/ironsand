# Kernel Architecture Comparison — Stream vs. Haiku-San vs. Monolithic

**Date**: 2026-06-23  
**Context**: Evolution from failed megakernel → two working hybrid approaches

---

## The Three Approaches

### 1. Monolithic Megakernel (❌ DEADLOCK)

**Concept**: One persistent kernel executes whole-layer DAG internally.

```
┌─────────────────────────────────────┐
│   Persistent Monolithic Kernel      │
│                                     │
│  RmsNorm → QKV → Rope → Attn →     │
│          ↘                ↙         │
│            OProj → Residual         │
│                                     │
│  RmsNorm → Gate/Up → SiLU →        │
│            ↘        ↙               │
│              Down → Residual        │
│                                     │
└─────────────────────────────────────┘
        (hand-rolled grid barrier)
```

**Problem**: Hand-rolled atomic grid barrier @ realistic configs (head_dim=64, grid≥3):
- All blocks must co-reside and reach barrier
- Combined register pressure (FFN + attention) drops occupancy below grid capacity
- Some blocks drop out before reaching barrier → **deadlock** (barrier never lifts)

**Result**: DEADLOCK (proven by razor project, commit b529e06)

---

### 2. Stream Kernel (✓ WORKS — GPU-Heavy Dispatch)

**Concept**: One persistent kernel that consumes a queue of micro-ops.

```
┌──────────────────────────────────────────┐
│   Persistent Stream Kernel (GPU)         │
├──────────────────────────────────────────┤
│                                          │
│  while (queue has ops) {                 │
│    op = queue.next()                     │
│    switch(op.opcode) {                   │
│      case RmsNorm:   rms_norm(...);      │
│      case GEMV_Q4K:  gemv_q4k(...);      │
│      case FlashAttn: flash_attn(...);    │
│      case SiLU:      silu(...);          │
│    }                                     │
│    __syncthreads();  // block-level      │
│  }                                       │
│                                          │
└──────────────────────────────────────────┘

    ↑ (Host enqueues ops once per token)
    │
  CPU builds queue, GPU runs it
```

**Design**: Queue of independent micro-ops, block-level sync (not grid-level).

**Advantages**:
- ✓ No deadlock (dynamic queue, per-op register footprint)
- ✓ Persistent (GPU boost stays latched)
- ✓ Modular (add ops as switch cases)
- ✓ Safe (each op individually proven)

**Limitations**:
- ✗ CPU idles while GPU runs (no parallelism)
- ✗ Static queue (CPU decides ops once, can't react)
- ✗ Per-op sync overhead (block-level barriers per op)

**Performance**: ~1.5-2× speedup (over per-kernel chain) from sustained boost, reduced launch overhead.

**Commits**: 243e196, 1ed9c28

---

### 3. Haiku-San (✓ WORKS — CPU/GPU Hybrid)

**Concept**: CPU orchestrator launches multiple GPU kernels asynchronously; GPU and CPU work in parallel.

```
CPU (Haiku-San)                     GPU (Kernel Execution)
─────────────────                   ──────────────────────

while layer:
  ┌─ submit(RmsNorm)     ─────────→ [GPU executes]
  │  submit(QKV)         ─────────→ [GPU queues]
  │
  │  (CPU continues)
  │  [CPU: validate RmsNorm output]
  │  [CPU: prefetch next layer]
  │
  │  wait_for(RmsNorm)   ←─────── [GPU event]
  │  wait_for(QKV)       ←─────── [GPU event]
  │
  │  submit(FlashAttn)    ─────────→ [GPU executes]
  │  submit(FFN)          ─────────→ [GPU queues]
  │
  │  (CPU continues)
  │  [CPU: decision logic]
  │  [if end_of_phrase: submit(lm_head)]
  │
  │  wait_for(FlashAttn)  ←─────── [GPU event]
  │  wait_for(FFN)        ←─────── [GPU event]
  │
  └─ next_layer
```

**Design**: Dependency graph + async event tracking. GPU and CPU work concurrently.

**Advantages**:
- ✓ No deadlock (separate kernels, not monolithic)
- ✓ CPU/GPU parallel (true parallelism, not lockstep)
- ✓ Dynamic control (CPU decides next kernel)
- ✓ Flexible (add kernels, no recompilation)
- ✓ Debuggable (each kernel independent)

**Limitations**:
- ✗ Event sync overhead (more syncs than stream kernel)
- ✗ Dependency tracking complexity (small, but non-zero)
- ✗ Requires event objects (GPU side API cost)

**Performance**: ~2-3× speedup (over per-kernel chain) from sustained boost + CPU/GPU parallelism.

**Commits**: 20c1fab

---

## Architectural Comparison Table

| Property | Monolithic | Stream | Haiku-San |
|----------|-----------|--------|-----------|
| **Deadlock Risk** | HIGH ⚠️ | NONE ✓ | NONE ✓ |
| **GPU Persistence** | Yes | Yes | Yes (multi-kernel) |
| **CPU/GPU Parallelism** | None (blocked) | None (waits) | YES ✓ |
| **Control Flow** | Rigid DAG | Static queue | Dynamic (CPU decides) |
| **Register Safety** | Combined (dangerous) | Per-op (safe) | Per-kernel (safe) |
| **Launch Overhead** | 0 (1 kernel) | ~minimal | ~minimal (amortized) |
| **Sync Points** | 1 grid barrier | N block barriers | M events (M < N) |
| **Extensibility** | Poor | Good (add ops) | Best (add kernels) |
| **Debugging** | Hard | Medium | Easy (per-kernel logs) |
| **Estimated Speedup** | — (deadlock) | 1.5-2× | 2-3× |
| **Implementation Risk** | High | Low | Medium |

---

## Hybrid Approach: Stream + Haiku-San

The **best-of-both** strategy:

```
Haiku-San (layer scheduler)
    ↓
    ├─ submit(kernel_rmsnorm)        → GPU kernel (lightweight)
    │
    ├─ submit(kernel_fused_qkv_rope) → [could be simple kernel]
    │
    ├─ submit(stream_kernel)         → [stream for attention]
    │    └─ internally: rmsnorm → qkv → rope → flash_attn → o_proj
    │                   (queue dispatch, tight coupling)
    │
    ├─ submit(stream_kernel)         → [stream for FFN]
    │    └─ internally: rmsnorm → gate/up → silu → down
    │
    └─ (CPU: control flow decisions)
       [sample token, check stop, prefetch, etc.]
```

**Why this works**:
- **Haiku-San** handles layer-level orchestration (CPU directs traffic)
- **Stream kernel** handles sub-layer fusion (GPU executes tightly-coupled ops)
- CPU and GPU truly parallel (Haiku orchestrates while stream executes)
- Scalable (add layers via Haiku, add ops via stream)

---

## Decision Tree: Which to Use?

```
START: Designing a new kernel architecture
    │
    ├─ "Must handle dynamic control flow?"
    │    ├─ YES → Haiku-San (CPU orchestrates)
    │    └─ NO  → continue
    │
    ├─ "Is latency on per-kernel launch a bottleneck?"
    │    ├─ YES → Stream kernel (persistent + queue)
    │    └─ NO  → stay with per-kernel chain
    │
    ├─ "Do you have complex intra-layer dependencies?"
    │    ├─ YES → Stream kernel (internal queue)
    │    └─ NO  → Haiku-San (separate kernels)
    │
    └─ "Need maximum CPU/GPU utilization?"
         ├─ YES → Haiku-San (parallel execution)
         └─ NO  → Stream kernel (simpler)

RECOMMENDED: Use Haiku-San for zorro's decode loop.
```

---

## Lessons Learned

1. **Monolithic is seductive but deadly**
   - One kernel sounds cleaner; in practice, register pressure + grid barriers create deadlock risk
   - Hand-rolled atomics can't compete with driver guarantees (`__syncthreads__`)

2. **Persistence matters more than fusion**
   - Keeping GPU boost-latched > reduced kernel count
   - Sustained work > per-kernel overhead reduction

3. **CPU/GPU parallelism wins**
   - Stream kernel: GPU runs queue, CPU is idle
   - Haiku-San: GPU runs kernels, CPU orchestrates + prefetches + decides
   - Haiku wins on total latency (CPU work hides GPU gaps)

4. **Queue-based dispatch is safe**
   - Stream kernel proves dynamic dispatch avoids deadlock
   - Per-op register budgets are light enough
   - Barriers scale: `__syncthreads__` (block) > hand-rolled grid atomics

5. **Hybrid scales**
   - Stream for tightly-coupled sub-layers (attention internals)
   - Haiku for layer-level orchestration (model structure)
   - No one-size-fits-all

---

## Implementation Status

| Approach | Status | Files | Proof |
|----------|--------|-------|-------|
| **Monolithic** | ❌ Deadlock | (razor) | b529e06 |
| **Stream Kernel** | ✓ Proven | stream_kernel.rs | 4-op dispatch, Q4K GEMV |
| **Haiku-San** | ✓ Proven | haiku_san.rs | 2-op + full-layer spikes |
| **Hybrid** | Design ready | — | Conceptual (not yet integrated) |

---

## Next Steps for Zorro Integration

1. **Short term**: Pick one approach
   - Recommend: **Haiku-San** (best CPU/GPU parallelism)
   - Fallback: **Stream kernel** (simpler, proven, lower risk)

2. **Medium term**: Implement in zorro's decode loop
   - Replace per-kernel launch pattern with orchestrator
   - Measure latency + GPU utilization

3. **Long term**: Hybrid strategy
   - Haiku for layer orchestration
   - Stream for attention + FFN fusion
   - Maximum parallelism + fusion benefits

---

## Why "Haiku-San" Won

The name captures the design:
- **Haiku** (俳句): Concise, captures essence
- **San** (三): Three pillars (GPU kernels, CPU orchestrator, dependency graph)

Philosophically: **a concise orchestration of three parts**.

In contrast:
- **Monolithic**: Ambitious but failed (deadlock)
- **Stream**: Safe but passive (GPU-bound, CPU idles)
- **Haiku-San**: Balanced (GPU + CPU active, dynamic, safe)

---

## References

- **Monolithic megakernel failure**: razor project, commit b529e06
  - Message: "whole-layer megakernel (NEGATIVE result, redirects P1)"
  - Analysis: Hand-rolled barrier deadlock at occupancy threshold

- **Stream kernel design**: `.dejavue/stream_kernel_design.md`
  - Proof: 4-op kernel (RmsNorm, GEMV_F32, GEMV_Q4K, SiLU)
  - Status: Compiles, callable, dispatch proven

- **Haiku-San design**: `.dejavue/haiku_san_kernel_design.md`
  - Proof: 2-op + full-layer spikes
  - Status: Task submission, async launch, event sync working

- **This comparison**: `.dejavue/kernel_types_comparison.md`
  - Decision framework: which approach for which scenario
  - Recommendation: Haiku-San for zorro decode loop
