# Role-Based Decode Kernels

Specialized GPU kernels optimized for the **decode phase** of LLM inference (single-token generation).

## What's Different About Decode?

| Metric | Prefill (Batch) | Decode (1 Token) |
|--------|-----------------|------------------|
| **Input shape** | [batch_size, seq_len] | [1, seq_len] (KV cache) |
| **Attention computation** | Multi-query, batch ops | Single-query, cache reuse |
| **GEMV dimension** | Small batch (8-32 rows) | Single row (thin 8192→8192) |
| **RMS norm** | Vectorized over batch | L1-cache fit, ~32 KB |
| **Optimization focus** | Throughput, TFLOPs | Latency per token, memory reuse |

**Result**: Different kernels needed for different phases.

## Kernels (Phase 1: Decode)

### 1. RoleRMSNormSingle ✅ (Week 1, Days 1-3)

RMS normalization for a single 1×hidden vector.

```rust
pub fn role_rms_norm_single(
    input: *const f32,      // [hidden_dim]
    output: *mut f32,       // [hidden_dim]
    hidden_dim: u32,
    eps: f32,
) { ... }
```

**Performance Target**:
- Latency: <100 μs
- Memory: L1-cache fit (32 KB for 8192 dims)
- Occupancy: 100% (single block)

**CPU Reference**: `cpu_reference::rms_norm_single()`

**Why it's faster**:
- Batch kernels use vectorized operations across batch
- Single-row norm is simpler: no batch dimension
- Shared memory (warp reduces) instead of atomics

### 2. RoleGEMVDecodeSingle (Week 1, Days 4-5)

Thin matrix-vector product: [hidden × hidden] @ [hidden]

```rust
pub fn role_gemv_decode_single(
    matrix: *const f32,    // [hidden_dim, hidden_dim] (row-major)
    vector: *const f32,    // [hidden_dim]
    output: *mut f32,      // [hidden_dim]
) { ... }
```

**Performance Target**:
- Latency: <500 μs per layer (8192 × 8192)
- Memory coalescing: Optimal (row-major layout)
- Occupancy: >80%

**CPU Reference**: `cpu_reference::gemv_decode_single()`

**Why it's faster**:
- Batch GEMV handles multiple rows concurrently
- Single-row GEMV: only one row to output
- Cache-friendly (no scatter, perfect coalescing)

### 3. RoleFlashAttnSingle (Week 1, Days 6-7)

Single-query attention with cached key/value.

```rust
pub fn role_flash_attn_single(
    q: *const f32,         // [1, num_heads, head_dim]
    k_cache: *const f32,   // [seq_len, num_heads, head_dim]
    v_cache: *const f32,   // [seq_len, num_heads, head_dim]
    output: *mut f32,      // [1, num_heads, head_dim]
    seq_len: u32,
) { ... }
```

**Performance Target**:
- Latency: <2000 μs for 1024-token cache
- Memory: All reads from cache (no compute K/V)
- Occupancy: >85%

**CPU Reference**: `cpu_reference::flash_attn_single()`

**Why it's faster**:
- Batch attention computes QKV for multiple tokens
- Single-query attention: Q is fixed, K/V cached
- No computation of K or V (they're reused)

---

## Quick Start

### Build the kernels

```bash
cargo build -p role_kernels_decode
```

This compiles:
- Device code to PTX (NVIDIA GPU bytecode)
- Host-side launchers (Rust)
- CPU reference implementations (Rust, no GPU)

### Test with CPU reference

```bash
cargo test -p role_kernels_decode --lib cpu_reference
```

Validates correctness without GPU hardware.

### Benchmark on GPU

```bash
cargo run --example role_kernels_decode_demo --release
```

Measures latency on your GPU.

---

## Integration with Haiku-San

Each kernel is dispatched by opcode:

```rust
use haiku_san::HaikuSan;
use role_kernels_decode::{OP_RMSNORM_SINGLE, OP_GEMV_DECODE_SINGLE};

let mut orchestrator = HaikuSan::new();

// Submit RMS norm kernel
let task1 = orchestrator.submit_task(
    "RmsNorm",
    OP_RMSNORM_SINGLE,
    1,           // single row
    hidden_dim,  // hidden dimension
);

// Submit GEMV kernel
let task2 = orchestrator.submit_task(
    "GEMV_Decode",
    OP_GEMV_DECODE_SINGLE,
    hidden_dim,  // output rows
    hidden_dim,  // input cols
);

// Declare dependency: task2 waits for task1
orchestrator.add_dependency(task2, task1);

// Launch all async
orchestrator.launch_all_async(&stream)?;
```

---

## Opcodes (Haiku-San Dispatcher)

```rust
const OP_RMSNORM_SINGLE: u32 = 30;
const OP_GEMV_DECODE_SINGLE: u32 = 31;
const OP_FLASH_ATTN_SINGLE: u32 = 32;
```

When Haiku-San sees opcode 30, it calls the RMS norm kernel.
When it sees opcode 31, it calls the GEMV kernel.
Etc.

---

## Correctness Validation

Each kernel is validated via **L2-rel error** against CPU reference:

```
L2-rel = ||GPU_output - CPU_output||_2 / ||CPU_output||_2
```

**Acceptance criteria**: L2-rel < 1e-4

This threshold accounts for:
- Float32 precision (1e-6 baseline)
- Rounding differences (1-2 ULPs)
- Numerical stability (small epsilon)

---

## Performance Characterization

### Week 4 Benchmarking

Goal: Verify latency targets on actual GPU.

```
Kernel                  | Target  | Measured (NVIDIA 5070 Ti)
────────────────────────┼─────────┼──────────────────────────
RoleRMSNormSingle       | <100 μs | ??? (to be measured)
RoleGEMVDecodeSingle    | <500 μs | ??? (to be measured)
RoleFlashAttnSingle     | <2000 μs| ??? (to be measured)
```

### Expected Decode Speedup

With all three decode roles integrated into zorro:

```
Baseline (per-kernel dispatch):   6 ms/token
With decode roles:                5 ms/token
Speedup:                          20%
```

---

## Design Documents

See `doc/ROLE_KERNELS_DESIGN.md` for:
- Why decode kernels differ from prefill
- Register pressure analysis
- Memory bandwidth utilization
- Scaling to longer caches

---

## Files

### Device Code (GPU)
- `kernels/src/lib.rs` — CUDA kernels (PTX)

### Host Code (CPU/GPU Interface)
- `src/lib.rs` — Launchers, opcode definitions
- `src/cpu_reference.rs` — CPU reference implementations (for testing)

### Build
- `build.rs` — Compiles kernels to PTX
- `Cargo.toml` — Dependencies (cust, haiku_san, cuda_std)

---

## Next Steps

1. **Week 1, Days 1-3**: Implement + test RoleRMSNormSingle ← **YOU ARE HERE**
2. **Week 1, Days 4-5**: Implement + test RoleGEMVDecodeSingle
3. **Week 1, Days 6-7**: Implement + test RoleFlashAttnSingle
4. **Week 2**: Integrate all three into Haiku-San
5. **Week 3**: Wire into zorro decode loop
6. **Week 4**: Measure 20% decode speedup

---

## Debugging Tips

### Compilation Fails
- Check CUDA path: `echo $CUDA_PATH`
- Check LLVM 19: `which llvm-config-19` or `$LLVM_CONFIG_19`
- Rebuild from scratch: `cargo clean && cargo build`

### Tests Fail (CPU Reference)
- Check CPU implementation in `cpu_reference.rs`
- Run with `RUST_LOG=debug` for detailed output
- Compare against hand-calculated values

### GPU Tests Fail (Correctness)
- Check L2-rel error: should be <1e-4
- Verify input shapes and allocations
- Check GPU memory availability
- Use `cuda-memcheck` if available

### Performance Below Target
- Profile with `nsys` or `nvvp` (NVIDIA tools)
- Check occupancy (should be >80%)
- Check memory bandwidth (GB/s vs peak)
- Consider register pressure (may need tuning)

---

## License

MIT or Apache-2.0 (same as ironsand).
