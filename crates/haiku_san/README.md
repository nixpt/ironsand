# Haiku-San: CPU/GPU Hybrid Orchestrator

Lightweight CPU-side task orchestrator for distributed GPU kernel chains in inference workloads.

## What's This For?

When you can't fit a whole model layer into a single monolithic GPU kernel (register pressure, occupancy cliffs, hand-rolled synchronization deadlock), you need a safe way to orchestrate multiple smaller kernels on the GPU while keeping the CPU and GPU busy in parallel.

**Haiku-San solves that:**
- ✅ Submit 64 tasks per token without blocking
- ✅ GPU executes in dependency order (your DAG)
- ✅ CPU prefetches, validates, makes decisions while GPU runs
- ✅ No deadlock (uses GPU driver's event sync, not hand-rolled atomics)

## Quick Start

```rust
use haiku_san::HaikuSan;
use cust::stream::Stream;

let mut orchestrator = HaikuSan::new();

// Submit tasks (no GPU launch yet)
let task1 = orchestrator.submit_task("RmsNorm", 0, 256, 256);
let task2 = orchestrator.submit_task("GEMV", 1, 512, 256);

// Declare dependencies (task2 waits for task1)
orchestrator.add_dependency(task2, task1);

// Launch all async (GPU starts executing; CPU continues)
orchestrator.launch_all_async(&stream)?;

// CPU work in parallel while GPU executes
prefetch_next_layer_weights();
validate_previous_output();

// Synchronize when ready
orchestrator.wait_for(task2)?;
```

## Architecture

### Why Not Just One Big Kernel?

```
Monolithic Approach:
  ┌─────────────────────────────────────────┐
  │ [RmsNorm → QKV → Rope → FlashAttn → OProj]
  │ [RmsNorm → GateUp → SiLU → Down]        │ ← ALL in one kernel
  └─────────────────────────────────────────┘
  
  Problem: Register pressure → low occupancy → some blocks idle before barrier
           → hand-rolled atomics deadlock ❌
```

```
Haiku-San Approach:
  CPU submits 64 small tasks to GPU queue:
    [Task1: RmsNorm] → [Task2: QKV] → [Task3: FlashAttn] → ...
  
  GPU executes in dependency order (no deadlock, safe barriers) ✅
  CPU prefetches weights while GPU runs ✅
```

### Key Design

- **Task**: A small kernel (RmsNorm, GEMV, SamplerSingle, etc.) with opcode, dimensions
- **Queue**: CPU-side submission queue (not GPU-resident)
- **Dependencies**: `task_B depends_on task_A` → GPU event synchronization
- **Orchestration**: CPU doesn't wait between submissions → GPU sees all tasks at once
- **Parallelism**: GPU executes while CPU prefetches, validates, loads cache

## Capacity Analysis

**Why 64 kernels per token?**

| Metric | Value | Constraint |
|--------|-------|-----------|
| GPU events available | ~10,000 | Soft cap before driver overhead |
| CPU overhead per task | ~5 μs | Submission, event tracking |
| GPU execution time per task | ~100 μs | Per-kernel kernel launch overhead |
| CPU budget per token | ~300 μs | <10% of GPU execution |
| Safe kernel count | 64 | 300 μs ÷ 5 μs/task |
| GPU utilization | 85% | with 64 kernels |

See `doc/CAPACITY_ANALYSIS.md` for detailed math.

## API Reference

### `HaikuSan::new() -> Self`
Create a new orchestrator.

### `submit_task(name, opcode, m, n) -> TaskId`
Queue a kernel task (not launched yet).
- `name`: Human-readable debug name
- `opcode`: Kernel selector (e.g., 0=RmsNorm, 1=GEMV_F32)
- `m`: Output size (rows, tokens, etc.)
- `n`: Input size (columns, hidden_dim, etc.)

### `add_dependency(dependent, task_id)`
Task `dependent` waits for `task_id` to complete.

### `launch_all_async(stream) -> Result<()>`
Submit all queued tasks to GPU (returns immediately).

### `wait_for(task_id) -> Result<()>`
Synchronize CPU on GPU event for this task.

### `stats() -> OrchestrationStats`
Get submission/completion counts and timing.

## Opcodes (Kernel Selector)

Standard opcodes for hybrid inference:

```rust
const OP_RMSNORM: u32 = 0;
const OP_GEMV_F32: u32 = 1;
const OP_ACTIVATION_SILU: u32 = 2;
const OP_GEMV_Q4K: u32 = 3;

// Stream block kernels (composite)
const OP_STREAM_ATTN_BLOCK: u32 = 10;
const OP_STREAM_FFN_BLOCK: u32 = 11;
```

Role-based kernels add new opcodes:

```rust
// Prefill roles
const OP_RMSNORM_BATCH: u32 = 20;
const OP_GEMV_BATCH_PREFILL: u32 = 21;
const OP_FLASH_ATTN_BATCH: u32 = 22;

// Decode roles
const OP_RMSNORM_SINGLE: u32 = 30;
const OP_GEMV_DECODE_SINGLE: u32 = 31;
const OP_FLASH_ATTN_SINGLE: u32 = 32;
const OP_SAMPLER_SINGLE: u32 = 33;
```

## Testing

Run basic unit tests:
```bash
cargo test -p haiku_san
```

For full GPU spike tests, see `examples/attn` which uses Haiku-San in action.

## Design Documents

- **HAIKU_SAN_DESIGN.md** — Core orchestration architecture and why it avoids deadlock
- **CAPACITY_ANALYSIS.md** — Mathematical analysis of kernel capacity limits
- **ROLE_KERNELS.md** — Phase-aware kernel specialization (prefill vs decode)
- **HYBRID_EXTENSIONS.md** — Pipeline parallelism, token streaming, speculation

## Next Steps

### Phase 1: Role-Based Kernels (2 weeks)
Implement decode-optimized kernels:
- `RoleRMSNormSingle` (<100 μs)
- `RoleGEMVDecodeSingle` (<500 μs)
- `RoleFlashAttnSingle` (<2000 μs)

### Phase 2: Prefill Roles (2 weeks)
Batch-optimized kernels for prefill phase.

### Phase 3: Advanced
- Layer pipelining (concurrent layers)
- Token streaming (overlapped generation)
- Speculative execution (multi-token hypotheses)

## Attribution

Built on [Rust-GPU/rust-cuda](https://github.com/Rust-GPU/rust-cuda) (MIT/Apache-2.0).
See ATTRIBUTION.md in the repo root.

## License

MIT or Apache-2.0 (same as upstream rust-cuda).
