# Haiku-San Kernel Design — CPU/GPU Hybrid Orchestration

**Status**: Architecture + design (pre-implementation)  
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

## Architecture: Haiku-San (Three-Part System)

**三 (san) = three pillars:**

### 1. GPU Kernel Registry (Lightweight Library)
Each kernel is a **self-contained CUDA function**:
- `gemv_q4k(input, weights, output, m, n)`
- `rmsnorm(input, output, n, eps)`
- `flash_attn(q, k, v, output, l, s, heads)`
- `lm_head_sparse(hidden, logits, allowed_tokens)`
- `silu(input, output, n)`

No queue, no persistence — just kernels.

### 2. CPU Orchestrator (Haiku — Concise, ~500 lines Rust)
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
    pub dependencies: HashMap<u64, Vec<u64>>,  // task_id → [task_ids it needs]
}

impl HaikuSan {
    pub async fn submit(&mut self, task: KernelTask) {
        // Launch GPU kernel, store event, don't wait
        let event = gpu.launch(task);
        self.inflight.insert(task.id, event);
    }
    
    pub async fn wait_for(&mut self, task_id: u64) {
        // Block until this task's GPU event fires
        self.inflight[&task_id].synchronize()?;
    }
    
    pub async fn orchestrate_token(&mut self, token_idx: usize) {
        // Pseudocode:
        // 1. launch GEMV_Q4K (hidden → ffn_up)
        // 2. launch RmsNorm (ffn_up → attn_input) — ASYNC with GEMV
        // 3. wait_for(GEMV) — blocks CPU, allows GPU to catch up
        // 4. validate FFN energy
        // 5. launch FlashAttn (attn_input → attn_output)
        // 6. while GPU runs FlashAttn, CPU: check if end-of-phrase
        // 7. if end: launch lm_head_sparse (attn → logits)
        // 8. CPU: run sampler, emit token_id
    }
}
```

### 3. Dependency Graph (Data Structure)
Pre-computed at startup:
```
Layer 0:
  input → RmsNorm → qkv_proj → rope → flash_attn → o_proj → residual → output
         ↓
  FFN: rmsnorm → gate/up → silu → down → residual
  
Graph edges (what must finish before next kernel):
  rope(qkv) → flash_attn  (qkv must be ready)
  flash_attn → o_proj     (attn output ready)
  gate/up → silu          (FFN gate/up ready)
```

Data structure: adjacency list or topological sort.

---

## Why This Works (vs. Alternatives)

| Metric | Monolithic | Stream | Haiku-San |
|--------|-----------|--------|-----------|
| **Deadlock risk** | HIGH | NONE | NONE |
| **Launch overhead** | 0 (1 kernel) | ~minimal (1 queue) | ~minimal (GPU-side async) |
| **GPU/CPU parallelism** | None (GPU-bound) | None (CPU builds queue) | **YES** (CPU orchestrates async) |
| **Control flow** | Hard (DAG-locked) | Hard (queue is static) | **Easy** (CPU decides) |
| **Register pressure** | Combined (dangerous) | Per-op (safe) | Per-op (safe) |
| **Scalability** | Poor (1 kernel) | Good (add ops) | **Best** (add kernels) |
| **Debugging** | Hard (monolith) | Medium (queue) | **Easy** (per-kernel logs) |

**The win**: CPU isn't idle while GPU runs. CPU can:
- Validate outputs (data sanity checks)
- Decide next kernel (control flow)
- Prepare next data (prefetch, rearrange)
- Run other work (sampler, token feedback)

---

## Implementation Strategy

### Phase 1 — Mechanism Proof (Spike)
Use **2 simple GPU kernels** + CPU async orchestrator:

```rust
// GPU side:
fn gemv_q4k(input, weights, output) { ... }
fn silu(input, output) { ... }

// CPU side (haiku_san.rs):
pub fn orchestrate_two_op() {
    task1_id = gpu.launch_async(gemv_q4k, ...);  // don't wait
    task2_id = gpu.launch_async(silu, ...);      // don't wait
    
    orchestrator.wait_for(task1);  // only sync when needed
    orchestrator.wait_for(task2);
    
    // Parallel phase: while GPU runs, CPU does work
    while gpu.inflight(task1_id) {
        cpu_validate_output(task1);  // check data
        // ...
    }
}
```

### Phase 2 — Full Decode Loop
Chain all per-layer kernels:
```
for layer in layers:
    submit(rmsnorm_1) → 
    submit(qkv_proj) →
    submit(rope) →
    [CPU: validate qkv]
    submit(flash_attn) →
    [CPU: prefetch next layer weights?]
    submit(o_proj) →
    submit(rmsnorm_2) →
    submit(ffn_gate_up) →
    [CPU: check phrase boundary?]
    submit(silu) →
    submit(ffn_down) →
    wait for layer done
```

### Phase 3 — Control Flow (Conditional Kernels)
```
submit(lm_head_sparse)  // always
[CPU: sample token, check stop condition]
if token == stop:
    return
else:
    submit(embedding_gather)
    continue decode
```

---

## Comparison: Haiku-San vs. Stream Kernel

**Stream Kernel** (from earlier spike):
- ✓ One persistent GPU kernel
- ✓ Consumes queue of micro-ops
- ✗ CPU marshals queue once, GPU runs it
- ✗ CPU idles while GPU executes

**Haiku-San**:
- ✓ Multiple lightweight GPU kernels
- ✓ CPU orchestrates async launches
- ✓ CPU/GPU parallel execution
- ✓ CPU can react to data (control flow)
- ✗ More synchronization points (events)
- ✗ More complex scheduling logic

**Hybrid approach**: Use **Stream kernel for heavy ops** (e.g., multi-op fusion like attention+MLP), orchestrated by **Haiku-San** at the layer level.

```
Haiku-San (layer scheduler)
    ↓
    submit(rmsnorm) → GPU kernel
    submit(fused_qkv_rope) → [could be stream kernel if complex]
    submit(flash_attn) → stream kernel [internal: rmsnorm→qkv→attn→o]
    submit(fused_ffn) → stream kernel [internal: gate/up→silu→down]
    [CPU: decide next layer]
```

---

## Integration with Zorro + Razor

### Zorro Path
```rust
// zorro's decode loop becomes:

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

### Razor Path
Haiku-San IS the hypervisor concept from razor's design:
- GPU side: persistent kernels (stream kernel)
- CPU side: lightweight scheduler (haiku-san)
- Handshake: GPU events, CPU control flow

Razor's §6e ("device-VM + hypervisor") = this.

---

## Benefits

1. **Parallelism**: CPU works while GPU runs (2-3× CPU utilization)
2. **Flexibility**: CPU decides next kernel (e.g., early exit, dynamic batch)
3. **Safety**: Per-kernel register budgets (no combined pressure)
4. **Debuggability**: Each kernel independent, testable in isolation
5. **Scalability**: Add kernels without changing orchestrator
6. **Portability**: Haiku-San logic is device-agnostic (GPU events are standard)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| **Event sync overhead** | Only sync at data dependencies, not every op |
| **CPU→GPU latency** | Pre-queue kernels; GPU ahead of CPU execution |
| **Stale outputs** | Track task IDs, validate before use |
| **Deadlock if CPU waits on GPU that waits on CPU** | Don't. CPU never blocks GPU; GPU doesn't block CPU. |

---

## Relationship to Stream Kernel + Megakernel

```
              Monolithic       Stream         Haiku-San
              ──────────       ──────         ──────────
GPU arch      1 big kernel     1 kernel       N kernels
              (deadlock)       (queue)        (distributed)

CPU role      Idle             Build queue    Orchestrate
              (blocked)        (once/token)   (async, reactive)

Launch cost   0                ~0              ~0 (amortized)

GPU/CPU sync  Async            Sync once       Events (minimal)

Control flow  Static DAG       Static queue   Dynamic (CPU decides)
```

This is the **middle path**: lighter than monolithic, more flexible than stream, more parallel than per-kernel chain.

---

## Files (When Built)

- `examples/attn/kernels/src/haiku_san.rs` — orchestrator (Rust, CPU-side)
- `examples/attn/kernels/src/kernels_registry.rs` — individual GPU kernels
- `examples/attn/src/main.rs` — haiku_san + multi-kernel decode test

---

## Next Steps

1. **Spike**: Implement haiku_san with 2 kernels (GEMV_Q4K → SiLU)
2. **Measure**: Compare latency (per-kernel vs. orchestrated) on 5070 Ti
3. **Extend**: Add all per-layer kernels, measure decode speedup
4. **Profile**: CPU utilization (should be 50%+ while GPU runs)
5. **Integrate**: Swap zorro's decode loop for haiku_san version

---

## Why "Haiku-San"?

- **Haiku** (俳句): Brief, concise; captures essence in few syllables
- **San** (三): Three — pillars (GPU kernels, CPU orchestrator, dependency graph)
- **Haiku-San** (俳句-三): Concise orchestration of three parts

Poetic name for a pragmatic design. 🎋
