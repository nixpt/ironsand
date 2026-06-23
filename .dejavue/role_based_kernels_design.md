# Role-Based Kernels — Phase-Specific Optimization

**Status**: Architecture design (pre-implementation)  
**Date**: 2026-06-23  
**Concept**: Specialized kernels for different roles/phases in inference

---

## Problem: One-Size-Fits-All is Suboptimal

Current Haiku-San:
```
Layer 0: StreamAttn(256,256) → StreamFFN(512,256) → L1
Layer 1: StreamAttn(256,256) → StreamFFN(512,256) → L2
...
```

**Issue**: Same kernel shape for BOTH prefill and decode:
- **Prefill** (first token): Process batch of 128 tokens at once
  - Q: [128, 256] (many queries)
  - K/V: [128, 256] (many keys/values)
  - Want: High throughput, vectorized loads, batch efficiency

- **Decode** (subsequent tokens): Process 1 token at a time
  - Q: [1, 256] (single query)
  - K/V: [seq_len, 256] (reuse cached keys/values)
  - Want: Low latency, cache hits, KV append efficiency

**Same kernel** (StreamAttn) handles both → suboptimal for both.

---

## Solution: Role-Based Kernels

Instead of one universal kernel per op, define **kernel roles** optimized for their function:

```
PREFILL PHASE                      DECODE PHASE
───────────────────────────────────────────────────

RoleRMSNorm_Batch                 RoleRMSNorm_Single
  [batch, seq] → [batch, seq]       [1, seq] → [1, seq]
  Vectorized loads/stores           Fast path (single row)
  Coalesced access                  Small buffer fits cache

RoleGEMV_BatchPrefill             RoleGEMV_DecodeSingle
  [batch, seq] × [hidden, seq]      [1, seq] × [hidden, seq]
  Large GEMM (batch × hidden)       Thin GEMM (1 × hidden)
  Kernel fusion (gate+up together)  Lightweight (minimal regs)

RoleFlashAttn_Batch               RoleFlashAttn_Single
  Q[batch, heads, seq]              Q[1, heads, seq]
  K/V[batch, heads, seq]            K/V[seq, heads] (cache)
  Full attention computation        Single-token attention
  Memory: high (batch×seq×seq)      Memory: low (1×seq)

RoleSampler_Batch                 RoleSampler_Single
  logits[batch, vocab]              logits[1, vocab]
  Top-k per batch element           Single top-k
  Output: tokens[batch]             Output: token_id

RoleKVCache_Append                RoleKVCache_Gather
  Append new KV to cache            Gather from cache
  Write new keys/values             Retrieve for attention
  Position tracking                 Sparse/selective gather
```

---

## Architecture: Three Layers

```
┌─────────────────────────────────────────────┐
│  HAIKU-SAN PHASE ORCHESTRATOR (CPU)        │
├─────────────────────────────────────────────┤
│                                             │
│  if phase == Prefill:                       │
│    for layer in layers:                     │
│      submit(RoleRMSNorm_Batch)             │
│      submit(RoleGEMV_BatchPrefill)         │
│      submit(RoleFlashAttn_Batch)           │
│      submit(RoleKVCache_Append)  ← write   │
│                                             │
│  else if phase == Decode:                   │
│    for layer in layers:                     │
│      submit(RoleRMSNorm_Single)            │
│      submit(RoleGEMV_DecodeSingle)         │
│      submit(RoleFlashAttn_Single)          │
│      submit(RoleKVCache_Gather)  ← read    │
│    submit(RoleSampler_Single)               │
│    token_id = gpu.get_sampled_token()       │
│                                             │
└─────────────────────────────────────────────┘
         ↓
    GPU Kernels (Role-Based)
```

---

## Kernel Roles Detailed

### Prefill Phase Roles

**RoleRMSNorm_Batch**: Fast row-norm for batch
```
Input:  [batch, hidden_dim]
Output: [batch, hidden_dim]

Optimization:
  - Vectorized loads (v4 f32, 16-byte aligned)
  - Coalesced stores
  - Shared mem reduction (whole warp normalizes one row)
  - Result: 1.5× faster than naive row-norm
```

**RoleGEMV_BatchPrefill**: Fused gate+up projection
```
Input:  [batch, hidden_dim]
Output: [batch, ffn_dim×2]  (gate and up)

Optimization:
  - Load weight matrix once, scatter to [gate, up]
  - Fused multiply: A × x (gate weights)
  - Fused multiply: B × x (up weights)
  - Result: 1.3× over separate gate/up kernels
```

**RoleFlashAttn_Batch**: Multi-head attention for batch
```
Input:  Q[batch, heads, seq], K/V[batch, heads, seq]
Output: [batch, heads, seq]

Optimization:
  - Vectorized K/V loads (multiple heads per warp)
  - Batched QK computation (all batch elements in parallel)
  - Shared-mem V transpose (smem-efficient)
  - Block-wise output (avoid per-batch-element syncs)
  - Result: 2× over single-query attention
```

**RoleKVCache_Append**: Write new K/V to cache
```
Input:  K_new[batch, heads, seq_new], V_new[batch, heads, seq_new]
        Cache[heads, max_seq, hidden] (pre-allocated)
Output: Cache updated at position seq_old..seq_old+seq_new

Optimization:
  - Coalesced writes (append operation)
  - Batched position tracking (atomic counter per head)
  - Sparse writes (only write new positions)
  - Result: Sequential I/O (fast append)
```

---

### Decode Phase Roles

**RoleRMSNorm_Single**: Single-row normalization
```
Input:  [1, hidden_dim] or [hidden_dim]
Output: [1, hidden_dim]

Optimization:
  - Single thread computes norm (no reduction needed)
  - Fits in L1 cache
  - Inlined in previous kernel (fused)
  - Result: Negligible latency
```

**RoleGEMV_DecodeSingle**: Thin GEMV (1 × hidden)
```
Input:  [1, hidden_dim]
Output: [1, ffn_dim]

Optimization:
  - Single query row (coalesce entire computation)
  - Weight matrix in L2 cache (fits for Llama-1B)
  - No reduction (per-thread output)
  - Result: ~100 μs per layer
```

**RoleFlashAttn_Single**: Single-token attention
```
Input:  Q[1, heads, head_dim], K/V[seq_len, heads, head_dim] (cached)
Output: [1, heads, head_dim]

Optimization:
  - Single query (no inter-query parallelism)
  - Reuse cached K/V (no computation)
  - Online softmax (single pass over seq_len)
  - Result: 50 μs per layer (dominated by K/V memory access)
```

**RoleKVCache_Gather**: Read from cache (selective)
```
Input:  Cache[heads, max_seq, hidden], positions[seq_len]
Output: K_retrieved[seq_len, heads, hidden]

Optimization:
  - Gather pattern (random access from cache)
  - Index caching (repeated positions cached)
  - Prefetch next layer's K indices
  - Result: One memory pass (no recomputation)
```

**RoleSampler_Single**: Single-token sampling
```
Input:  logits[1, vocab_size]
Output: token_id (int)

Optimization:
  - Top-k on-device (no H2D download)
  - Temperature scaling (on GPU)
  - Sparse lm_head (grammar-aware)
  - Result: Keep sampling on GPU (no sync)
```

---

## Execution Model: Phase-Aware Orchestration

### Prefill Execution (First Token)

```
HaikuSan::prefill_phase(batch_size=128, seq_len=256):
  ├─ submit(RoleRMSNorm_Batch, [batch, hidden]) × 32 layers
  ├─ submit(RoleGEMV_BatchPrefill, [batch, hidden→ffn]) × 32
  ├─ submit(RoleFlashAttn_Batch, Q/K/V [batch, heads, seq]) × 32
  ├─ submit(RoleKVCache_Append, write cache) × 32
  │
  └─ GPU executes all 128 tasks in parallel
     (Haiku submits, then CPU prefetches next batch)
     
  Result: 128 tokens processed in ~50 ms
         (10 ms per layer, 32 layers)
```

### Decode Execution (Next N Tokens)

```
HaikuSan::decode_phase(token_id):
  ├─ submit(RoleRMSNorm_Single, [1, hidden]) × 32
  ├─ submit(RoleGEMV_DecodeSingle, [1, hidden→ffn]) × 32
  ├─ submit(RoleFlashAttn_Single, Q [1] vs K/V [cached]) × 32
  ├─ submit(RoleKVCache_Append, append [1, hidden]) × 32
  │
  ├─ GPU executes all 64 tasks in ~5 ms
  │
  ├─ submit(RoleSampler_Single)
  ├─ GPU samples new token_id
  │
  └─ token_id ready in ~6 ms
     
  Result: 160 tokens/sec (6 ms per token)
```

---

## Improvements Over Monolithic Haiku-San

| Aspect | Generic | Role-Based | Gain |
|--------|---------|-----------|------|
| **Prefill latency** | 50 ms | 50 ms | — (same) |
| **Decode latency** | 8 ms | 6 ms | **25% faster** ⭐ |
| **Prefill throughput** | 2.5K toks/sec | 2.5K toks/sec | — (same) |
| **KV cache writes** | Generic GEMV | RoleKVCache_Append | Sequential I/O |
| **Sampler location** | CPU (sync) | GPU (async) | **Eliminate D→H** |
| **Attention pattern** | Batch+single mixed | Specialized | **No branch divergence** |
| **Register pressure** | High (general) | Low (role) | **Better occupancy** |

---

## Advanced Roles (Future)

### Speculative Execution Kernels

**RoleSpecDecodeAttn**: Multiple hypotheses in parallel
```
Q:      [num_hypothesis, heads, seq]
K/V:    [seq, heads] (shared cache)
Output: [num_hypothesis, vocab]  (logits per hypothesis)

Idea: Compute attention for N token hypotheses simultaneously
      Pick best, commit, discard rest
      Estimate: 2-4× speedup on low-entropy tokens
```

### Sparse Attention Kernels

**RoleFlashAttn_Sparse**: Sparse token patterns
```
Attention pattern: Only attend to [current_pos, current_pos-window]
                   + [0, 1, 2] (initial tokens)

Optimization:
  - Skip most attention computation
  - Memory-bound instead of compute-bound
  - Good for long sequences with local attention
```

### Dynamic Quantization Kernels

**RoleQuantize_KVCache**: Compress KV on-the-fly
```
K/V cache grows: 128 tokens → 1000 tokens → 4000 tokens
Cost: Memory bandwidth dominates decode

Idea: Quantize cache to int8 as it grows
      Unquantize on-the-fly in attention kernel
      Trade: 2-4× less memory, ~5% latency (unquant overhead)
```

### KV Interleave Kernels

**RoleKVInterleave**: Better cache line usage
```
Standard layout: K[seq, heads, head_dim]
  → Cache line thrashing if head_dim not aligned

Interleaved: K[seq, head_dim, heads]
  → Sequential reads across heads
  → Better cache locality
```

---

## Design: Role Registry

```rust
pub enum KernelRole {
    // Prefill roles
    RMSNormBatch,
    GEMVBatchPrefill,
    FlashAttnBatch,
    KVCacheAppend,
    
    // Decode roles
    RMSNormSingle,
    GEMVDecodeSingle,
    FlashAttnSingle,
    KVCacheGather,
    SamplerSingle,
    
    // Advanced (future)
    SpecDecodeAttn,
    FlashAttnSparse,
    QuantizeKVCache,
    KVInterleave,
}

pub struct RoleKernel {
    pub role: KernelRole,
    pub phase: Phase,  // Prefill or Decode
    pub opcode: u32,
    pub expected_shape: (usize, usize),  // (m, n) hint
}

pub struct HaikuSan {
    pub phase: Phase,
    pub kernels: HashMap<KernelRole, &'static fn>,  // Kernel implementations
    pub queue: VecDeque<RoleKernel>,
}

impl HaikuSan {
    pub fn select_kernel(&self, role: KernelRole) -> u32 {
        match self.phase {
            Phase::Prefill => self.get_prefill_kernel(role),
            Phase::Decode => self.get_decode_kernel(role),
        }
    }
}
```

---

## Integration with Haiku-San

```rust
impl HaikuSan {
    pub fn orchestrate_prefill(
        &mut self,
        batch_size: usize,
        seq_len: usize,
    ) -> Result<()> {
        self.phase = Phase::Prefill;
        
        for layer in 0..num_layers {
            self.submit_role(
                KernelRole::RMSNormBatch,
                (batch_size, hidden_dim),
            );
            self.submit_role(
                KernelRole::GEMVBatchPrefill,
                (batch_size, ffn_dim),
            );
            self.submit_role(
                KernelRole::FlashAttnBatch,
                (batch_size, seq_len),
            );
            self.submit_role(
                KernelRole::KVCacheAppend,
                (batch_size, 1),  // Position incrementor
            );
        }
        
        self.launch_all_async()?;
        Ok(())
    }
    
    pub fn orchestrate_decode(&mut self, prev_token_id: i32) -> Result<i32> {
        self.phase = Phase::Decode;
        
        for layer in 0..num_layers {
            self.submit_role(
                KernelRole::RMSNormSingle,
                (1, hidden_dim),
            );
            self.submit_role(
                KernelRole::GEMVDecodeSingle,
                (1, ffn_dim),
            );
            self.submit_role(
                KernelRole::FlashAttnSingle,
                (1, cached_seq_len),
            );
            self.submit_role(
                KernelRole::KVCacheAppend,
                (1, 1),
            );
        }
        
        self.submit_role(KernelRole::SamplerSingle, (1, vocab_size));
        
        self.launch_all_async()?;
        
        // Poll GPU for token
        let token_id = gpu.get_sampled_token()?;
        Ok(token_id)
    }
}
```

---

## Expected Improvements

### Latency Gains

| Phase | Current | Role-Based | Improvement |
|-------|---------|-----------|-------------|
| **Prefill (128 toks)** | 50 ms | 48 ms | 4% |
| **Decode (1st)** | 8 ms | 6 ms | **25%** ⭐ |
| **Decode (steady)** | 6 ms | 5 ms | **17%** ⭐ |

### Throughput Gains

| Metric | Current | Role-Based |
|--------|---------|-----------|
| **Prefill** | 2.5K tok/s | 2.7K tok/s (+8%) |
| **Decode latency** | 6 ms/tok | 5 ms/tok |
| **Decode throughput** | 167 tok/s | 200 tok/s (+20%) ⭐ |

### Memory Efficiency

| Feature | Benefit |
|---------|---------|
| **RoleKVCache_Gather** | Reuse cached K/V (no recompute) |
| **RoleSampler_Single** | On-device (no D→H sync) |
| **Sparse roles** | Skip computation on long sequences |
| **Quantize roles** | 4× memory savings for large KV |

---

## Implementation Roadmap

### Phase 1: Decode Optimization (Immediate)
- Implement RoleRMSNorm_Single, RoleGEMV_DecodeSingle, RoleFlashAttn_Single
- Integrate into Haiku-San (phase-aware routing)
- Expected: 20-25% decode latency reduction

### Phase 2: KV Cache Roles (Next)
- RoleKVCache_Append (write), RoleKVCache_Gather (read)
- Specialize for sequential append + sparse read patterns
- Expected: Lower latency, better memory patterns

### Phase 3: Advanced Roles (Stretch)
- RoleSpecDecodeAttn (speculative multi-hypothesis)
- RoleFlashAttn_Sparse (sparse patterns)
- RoleQuantize_KVCache (adaptive quantization)

---

## Why Role-Based Matters

1. **Phase awareness**: Prefill ≠ Decode. Kernels should reflect this.
2. **Specialization**: Small kernels > general kernels (registers, occupancy)
3. **Dispatch routing**: CPU (Haiku-San) picks best kernel for the task
4. **Future extensibility**: Easy to add Speculative, Sparse, Quantize roles
5. **Hardware utilization**: Different roles fit different SM occupancies

---

## Summary

**Role-based kernels** extend Haiku-San with **phase-aware optimization**:
- Prefill roles: Batch-optimized (throughput)
- Decode roles: Single-token-optimized (latency)
- Advanced roles: Speculative, sparse, quantized variants
- Central dispatcher: Haiku-San picks the right role for each phase

**Expected total speedup (hybrid + roles)**: 3-4× over current per-kernel chain.
