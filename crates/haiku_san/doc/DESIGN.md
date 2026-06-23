# Haiku-San Design: CPU/GPU Hybrid Orchestration

**Status**: Architecture + design (implementation in progress)  
**Date**: 2026-06-23  
**Concept**: Distributed kernels (GPU-heavy) orchestrated by lightweight CPU engine

---

## The Insight: Flip the Hierarchy

Current approaches:
- **Monolithic**: One kernel does everything (deadlock risk, register pressure)
- **Stream kernel**: GPU queue, CPU builds it once per token
- **Per-kernel chain**: 200 launches, GPU idles between them

**Haiku-San**: CPU orchestrates multiple GPU kernels **concurrently**:

```
     CPU (Haiku-San Orchestrator)          GPU (Kernel Execution)
     ─────────────────────────           ──────────────────────
     
     while token:
       ┌─ Submit GEMV_Q4K             → GPU executes
       │                                  (CPU continues)
       │  Check dependency graph
       │  ┌─ Submit RmsNorm next       → GPU queues
       │  │
       │  Wait for GEMV_Q4K complete  ← GPU signals
       │  Validate output
       │  ┌─ Submit FlashAttn          → GPU executes
       │  │  (with GEMV output)
       │  │
       │  Decision: is this token     
       │  the end of a phrase?        
       │  ┌─ Maybe: submit lm_head     → GPU
       │  │ Maybe: continue decode     → loop
       │  
       └─ Read final token
```

Key: **CPU and GPU work in parallel**, not in lockstep.

---

## Three Pillars

### 1. GPU Kernel Registry
Each kernel is a **self-contained CUDA function**:
- `gemv_q4k(input, weights, output, m, n)`
- `rmsnorm(input, output, n, eps)`
- `flash_attn(q, k, v, output, l, s, heads)`
- `silu(input, output, n)`

No queue, no persistence — just kernels.

### 2. CPU Orchestrator (Haiku)
Manages:
- **Dependency graph** (which kernel runs after which)
- **Async GPU launches** (submit, don't wait)
- **Data hazard tracking** (which output needed by next kernel)
- **Event synchronization** (minimal: only at data dependencies)
- **Control flow** (CPU decides next kernel based on outputs)

```rust
pub struct HaikuSan {
    pub queue: VecDeque<KernelTask>,
    pub inflight: HashMap<u64, Event>,  // task_id → GPU event
}

impl HaikuSan {
    pub fn submit(&mut self, task: KernelTask) {
        // Launch GPU kernel, store event, don't wait
        let event = gpu.launch(task);
        self.inflight.insert(task.id, event);
    }
    
    pub fn wait_for(&mut self, task_id: u64) {
        // Block until this task's GPU event fires
        self.inflight[&task_id].synchronize()?;
    }
}
```

### 3. Dependency Graph
Data structure defining which kernels depend on which:
```
Layer 0:
  input → RmsNorm → qkv_proj → rope → flash_attn → o_proj → output
  
  FFN: rmsnorm → gate/up → silu → down
  
Graph edges:
  rope(qkv) → flash_attn  (qkv must be ready)
  flash_attn → o_proj     (attn output ready)
  gate/up → silu          (FFN gate/up ready)
```

---

## Why This Works

| Metric | Monolithic | Stream | Haiku-San |
|--------|-----------|--------|-----------|
| **Deadlock risk** | HIGH | NONE | NONE |
| **GPU/CPU parallelism** | None | None | **YES** |
| **Control flow** | Hard | Hard | **Easy** |
| **Register pressure** | Combined | Per-op | Per-op |
| **Scalability** | Poor | Good | **Best** |
| **Debugging** | Hard | Medium | **Easy** |

**The win**: CPU isn't idle while GPU runs. CPU can:
- Validate outputs (sanity checks)
- Decide next kernel (control flow)
- Prepare next data (prefetch, rearrange)
- Run other work (sampler, token feedback)

---

## Capacity Analysis

**Why 64 kernels per token?**

| Metric | Value | Constraint |
|--------|-------|-----------|
| GPU events available | ~10,000 | Soft cap before driver overhead |
| CPU overhead per task | ~5 μs | Submission, event tracking |
| GPU execution time per task | ~100 μs | Per-kernel launch overhead |
| CPU budget per token | ~300 μs | <10% of GPU execution |
| Safe kernel count | 64 | 300 μs ÷ 5 μs/task |
| GPU utilization | 85% | with 64 kernels |

See `CAPACITY_ANALYSIS.md` for detailed math.

---

## Integration with Zorro

Haiku-San becomes zorro's decode loop:

```rust
let haiku = HaikuSan::new();
for token_idx in 0..max_tokens {
    for layer in &model.layers {
        haiku.submit(rmsnorm, layer.input);
        haiku.submit(qkv_proj, ...);
        // ...
        haiku.orchestrate_layer();
    }
    
    // CPU-side control flow:
    let token_id = gpu.sample_token();
    if model.should_stop(token_id) { break; }
}
```

---

## Relationship to Other Approaches

```
              Monolithic       Stream         Haiku-San
              ──────────       ──────         ──────────
GPU arch      1 big kernel     1 kernel       N kernels
              (deadlock)       (queue)        (distributed)

CPU role      Idle             Build queue    Orchestrate
              (blocked)        (once/token)   (async, reactive)

GPU/CPU sync  Async            Sync once      Events (minimal)

Control flow  Static DAG       Static queue   Dynamic (CPU decides)
```

This is the **middle path**: lighter than monolithic, more flexible than stream, more parallel than per-kernel chain.

---

## Benefits

1. **Parallelism**: CPU works while GPU runs (2-3× CPU utilization)
2. **Flexibility**: CPU decides next kernel (e.g., early exit, dynamic batch)
3. **Safety**: Per-kernel register budgets (no combined pressure)
4. **Debuggability**: Each kernel independent, testable in isolation
5. **Scalability**: Add kernels without changing orchestrator
6. **Portability**: Haiku-San logic is device-agnostic

---

## Why "Haiku-San"?

- **Haiku** (俳句): Brief, concise; captures essence in few syllables
- **San** (三): Three — pillars (GPU kernels, CPU orchestrator, dependency graph)
- **Haiku-San** (俳句-三): Concise orchestration of three parts

Poetic name for a pragmatic design. 🎋
