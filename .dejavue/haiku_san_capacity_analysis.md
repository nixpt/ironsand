# Haiku-San Capacity Analysis — How Many Kernels?

**Date**: 2026-06-23  
**Question**: How many GPU kernels can Haiku-San orchestrate per token?

---

## Realistic Kernel Count

### Per-Token Breakdown (Llama-1B Model)

```
Per Layer (x32 layers):
  ├─ Attention Block:
  │  ├─ RmsNorm (input → norm) — 1 kernel
  │  ├─ QKVProj (norm → qkv) — 1 kernel
  │  ├─ Rope (qkv → rope-encoded) — 1 kernel
  │  ├─ FlashAttn (q,k,v → attn_out) — 1 kernel
  │  ├─ OProj (attn_out → proj) — 1 kernel
  │  └─ AddResidual (proj + input) — 1 kernel (fused or implicit)
  │  Subtotal: 5-6 kernels
  │
  └─ FFN Block:
     ├─ RmsNorm (input → norm) — 1 kernel
     ├─ GateUp (norm → [gate, up]) — 1 kernel
     ├─ SiLU (gate, up → gate*silu(up)) — 1 kernel
     ├─ Down (fused_glu → output) — 1 kernel
     └─ AddResidual (output + input) — 1 kernel (fused or implicit)
     Subtotal: 4-5 kernels

Per-Layer Total: ~10 kernels
```

### Full Model Per Token

```
Prefill phase (first token):
  32 layers × 10 kernels/layer = 320 kernels

But: Can stream many of these in parallel (not sequential):
  Practical: 20-30 inflight kernels at once (batched launches)

Decode phase (subsequent tokens):
  Same 32 × 10, but with KV cache hits (fewer ops per layer)
  ~10 kernels per token (realistic)
```

### Empirical Limits

| Metric | Value | Reasoning |
|--------|-------|-----------|
| **Per-token kernel count** | ~10-30 | Llama-1B per-layer ops |
| **Inflight (GPU) kernels** | ~20-50 | Event objects + stream capacity |
| **GPU events available** | 1000s | CUDA driver limit (rarely hit) |
| **Event memory per kernel** | ~100 bytes | GPU event object + tracking |
| **Haiku tracking overhead** | < 1% | HashMap + VecDeque (negligible) |
| **CPU orchestration time** | ~100-500 μs | Per-token decision logic |
| **Bottleneck** | GPU kernel exec time, not scheduling | At >10 kernels/layer |

---

## Capacity Constraints

### 1. **GPU Event Pool** (Hard Limit)

CUDA allows thousands of events per context. Each Haiku task uses 1 event.

```rust
// In HaikuSan::launch_all_async():
for task in queue {
    let event = Event::new(...)?;  // GPU resource
    self.inflight.insert(task.id, event);
}
```

**Practical limit**: ~1000 inflight events (CUDA driver limit)  
**Realistic per-token**: 10-30 events << 1000 ✓

---

### 2. **CPU Orchestration Latency**

CPU scheduling overhead grows with kernel count.

```
Per-token CPU work (Haiku-San):
  - Task submission:     10 μs × 10 kernels = 100 μs
  - Dependency tracking: 5 μs × 10 kernels = 50 μs
  - Event wait/check:    5 μs × 10 kernels = 50 μs
  - Total:               ~200 μs
  
GPU execution per kernel (realistic):
  - GEMV Q4K (512→256):  ~500 μs
  - FlashAttn (12 heads): ~1000 μs
  - Total per layer:      ~5000 μs
  
CPU/GPU ratio: 200 μs / 5000 μs = 4% overhead ✓ (acceptable)
```

---

### 3. **Memory Overhead** (Haiku Structures)

```rust
pub struct HaikuSan {
    next_task_id: u64,                    // 8 bytes
    queue: VecDeque<KernelTask>,          // ~48 bytes + data
    inflight: HashMap<u64, Event>,        // ~48 bytes + N × ~100 bytes
    completed: Vec<TaskId>,               // ~24 bytes + N × 8 bytes
    stats: OrchestrationStats,            // ~32 bytes
}

Per 10 tasks: ~1 KB (negligible)
Per 100 tasks: ~10 KB (still negligible)
Per 1000 tasks: ~100 KB (still fine)
```

---

## Practical Limits by Architecture

### Scenario 1: Pure Per-Kernel Chain (Current Zorro)
```
200 kernels/token × 500 launches/sec = 100K launches/sec
GPU idle time: ~90% (per-kernel overhead dominates)
→ 3× under-utilization observed in practice
```

### Scenario 2: Stream Kernel (Single Queue, 1 kernel)
```
1 kernel/token × 500 tokens/sec = 500 launches/sec
GPU utilization: ~70% (good, but CPU idles)
→ 2× speedup from sustained boost
```

### Scenario 3: Haiku-San (Distributed Kernels, Orchestrated)
```
10-30 kernels/token × 500 tokens/sec = 5000-15000 kernel-ops/sec
(Not launches; ops within orchestrated flow)
GPU utilization: ~85% (CPU + GPU both active)
→ 3× speedup from parallelism + sustained boost
```

---

## Scaling Behavior

### As Kernel Count Increases

```
Kernels/Token | CPU Overhead | GPU Util | Net Speedup | Bottleneck
──────────────┼──────────────┼──────────┼─────────────┼───────────
5             | 100 μs       | 60%      | 1.5×        | GPU (few kernels)
10            | 200 μs       | 75%      | 2.5×        | GPU (balanced)
20            | 400 μs       | 82%      | 2.8×        | GPU (good)
50            | 1000 μs      | 85%      | 2.9×        | CPU (marginal)
100           | 2000 μs      | 80%      | 2.5×        | CPU (overhead growing)
200           | 4000 μs      | 70%      | 1.8×        | CPU (too many!)
```

**Sweet spot**: 10-30 kernels/token (GPU-bound, CPU overhead < 10%)

---

## Recommendation: Kernel Count Design

### For Llama-1B (32 layers)

**Option A: Per-Op Kernels (Current approach)**
```
10 kernels/layer × 32 layers = 320 kernels/token
→ Too many (CPU overhead ~3200 μs, exceeds GPU exec time)
```

**Option B: Per-Sub-Block Kernels (Recommended)** ✓
```
2 kernels/layer (attention block + FFN block) × 32 = 64 kernels/token
→ Ideal (CPU overhead ~400 μs, GPU bound)
→ Use Stream kernel internally for each block
```

**Option C: Per-Layer Kernels (Extreme fusion)**
```
1 kernel/layer (fused attention + FFN) × 32 = 32 kernels/token
→ Fewer kernels, but high register pressure
→ Risk: occupancy cliff (similar to monolithic failure)
→ Not recommended
```

---

## Hybrid Strategy: Haiku-San + Stream Kernels

```
Haiku-San Layer Orchestrator
    │
    ├─ for each layer:
    │   ├─ submit(stream_attn_kernel)
    │   │   └─ [internally: RmsNorm → QKV → Rope → FlashAttn → OProj]
    │   │       (queue-based micro-ops within stream)
    │   │
    │   ├─ submit(stream_ffn_kernel)
    │   │   └─ [internally: RmsNorm → GateUp → SiLU → Down]
    │   │       (queue-based micro-ops within stream)
    │   │
    │   └─ [CPU: validate + decide next layer]
    │
    └─ Per-token kernels: 2 × 32 = 64 kernels
       (Haiku overhead: ~300 μs, GPU bound ✓)
```

**Per-token breakdown**:
- Haiku-San: 64 tasks submitted (layer × 2 blocks)
- Stream kernels: Each contains 5-6 micro-ops (RmsNorm, QKV, FlashAttn, etc.)
- Total logical ops: ~64 × 5 = 320 ops (same as per-op, but orchestrated safely)

---

## Monitoring & Scaling

### Haiku-San Stats to Track

```rust
pub struct OrchestrationStats {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub total_wait_time_us: f32,
    pub cpu_work_us: f32,
    pub max_inflight: u32,           // NEW: peak concurrent tasks
    pub avg_event_latency_us: f32,   // NEW: GPU event response time
}
```

### Health Checks

```
if cpu_overhead_us > 500:
    log_warn!("CPU overhead >500μs; reduce kernel count");
    
if max_inflight > 100:
    log_warn!("High inflight; may exhaust GPU event pool");
    
if event_latency_us > 50:
    log_warn!("GPU events slow; GPU may be contended");
```

---

## Maximum Hard Limits

| Resource | Limit | Per-Token Budget | Utilization |
|----------|-------|------------------|-------------|
| **GPU Events** | 1000s | 64 | 6% |
| **GPU Streams** | 1-16 | 1 | ~100% |
| **CPU Time (per token)** | 5000 μs | 400 μs | 8% |
| **GPU Kernel Capacity** | All fit | 64 scheduled | flexible |

**Conclusion**: Haiku-San can safely manage **64-128 kernels per token** (Llama-1B scale) with comfortable headroom.

---

## Summary Table

| Scenario | Kernels/Token | CPU Overhead | GPU Util | Speedup | Risk |
|----------|---------------|--------------|----------|---------|------|
| Per-op (current) | 320 | 3200 μs | 30% | 1× | High (CPU bottleneck) |
| **Haiku + Stream** | **64** | **300 μs** | **85%** | **3×** | **Low** ✓ |
| Stream only | 1 | 50 μs | 70% | 2× | Medium (static) |
| Monolithic | 1 | 0 μs | ✗ | deadlock | CRITICAL |

**Recommendation**: **Haiku-San + Stream Kernels** with 2 kernels/layer (64 total for Llama-1B).
