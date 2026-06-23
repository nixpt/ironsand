# Hybrid Kernel Extensions — Advanced Patterns & Improvements

**Status**: Architecture extensions (design)  
**Date**: 2026-06-23  
**Context**: Building on role-based kernels for greater flexibility

---

## Five Major Extensions

### 1. Pipeline Parallelism (Layer Pipelining)

**Problem**: Today's model runs layers sequentially (L0 done → L1 starts)

**Idea**: Prefill multiple layers simultaneously (like data parallelism, but with layers)

```
Traditional (sequential):
  GPU: [L0 layer execution] [L1 layer execution] [L2 layer] ...
  CPU: waiting | waiting | waiting
  
Pipelined (layer-level parallelism):
  GPU: [L0] [L1] [L2] [L3] ...
       [L0][L1][L2][L3]  [L4][L5][L6][L7]   (next batch prefill)
       
  (Layer N of batch 1 while Layer N+k of batch 2)
```

**Implementation**:
```rust
impl HaikuSan {
    pub fn prefill_with_pipelining(&mut self, batches: Vec<Batch>) {
        // Submit multiple batches' layers concurrently
        for (batch_idx, batch) in batches.iter().enumerate() {
            for (layer_idx, layer) in batch.layers.iter().enumerate() {
                // Stagger: Batch 0 Layer 0, Batch 1 Layer 0, 
                //          Batch 0 Layer 1, Batch 1 Layer 1, ...
                let task_id = self.submit_role(
                    KernelRole::GEMVBatchPrefill,
                    (batch.size, ffn_dim),
                );
                self.add_dependency(
                    task_id,
                    previous_layer_same_batch(batch_idx, layer_idx - 1),
                );
            }
        }
        self.launch_all_async()?;
    }
}
```

**Expected gain**: 1.5-2× throughput on multi-batch prefill

---

### 2. Token Streaming (Out-of-Order Decode)

**Problem**: Wait for full sequence to generate before sampler runs

**Idea**: Stream tokens as they're generated, start decode of next token immediately

```
Traditional:
  GPU: [Layer 0] [Layer 1] ... [Layer 31] → Sampler
       ================================================
       (Total latency: ~30 ms)
       
Streamed:
  GPU: [Layer 0] → [Layer 1] → ... → [Layer 31] → Sampler → decode token[1]
       ↓            ↓                   ↓           ↓         (parallel)
       [L0-Token2]  [L1-Token2]  ...  [L31-Token2] Sampler2
       
  (First token latency: 30 ms)
  (Subsequent tokens: 1 ms as previous token streams through)
```

**Implementation**:
```rust
impl HaikuSan {
    pub async fn decode_streaming(&mut self, initial_token: i32) {
        let mut token_id = initial_token;
        let mut token_stream = tokio::sync::mpsc::channel(32);
        
        loop {
            // Submit layers (will generate next token)
            for layer in 0..num_layers {
                self.submit_role(RoleGEMVDecodeSingle, ...);
            }
            
            // Sampler is in the queue; it will fire when ready
            self.submit_role(RoleSamplerSingle, ...);
            
            // Don't wait; spawn decode of NEXT token
            let (next_tx, next_rx) = mpsc::channel();
            tokio::spawn(async move {
                self.launch_all_async().await;
                let token = gpu.get_sampled_token().await;
                next_tx.send(token).await;
            });
            
            // Meanwhile, CPU can:
            // - Validate token
            // - Update KV cache pointer
            // - Check stopping condition
            
            token_id = next_rx.recv().await?;
        }
    }
}
```

**Expected gain**: Reduced per-token latency (pipelining hides decode start cost)

---

### 3. Dynamic Batch Sizing (Adaptive Prefill)

**Problem**: Prefill batch size fixed (e.g., 128); some requests need smaller batches

**Idea**: Dynamically resize batch per request based on KV cache availability

```
Request queue: [ReqA:512] [ReqB:256] [ReqC:1024] ...
              
Dynamic batching:
  Prefill 1: [ReqA:512 + ReqB:256] = 768 tokens (fits)
  Prefill 2: [ReqC:1024] = 1024 tokens (exceeds single batch)
             → Auto-split into [ReqC-part1:512] + [ReqC-part2:512]
             
Haiku-San logic:
  if (current_batch_tokens + next_request_tokens) <= max_batch:
      add_to_batch(next_request)
  else:
      launch_current_batch()
      start_new_batch(next_request)
```

**Implementation**:
```rust
impl HaikuSan {
    pub fn add_request_to_prefill_batch(&mut self, req: Request) -> bool {
        if self.current_batch_size + req.seq_len <= self.max_batch_size {
            self.current_batch.push(req);
            true
        } else {
            // Current batch full; will be launched next
            false
        }
    }
    
    pub fn schedule_request(&mut self, req: Request) {
        loop {
            if self.add_request_to_prefill_batch(req.clone()) {
                break;  // Scheduled
            } else {
                // Wait for current batch to finish
                self.launch_prefill_batch().await;
            }
        }
    }
}
```

**Expected gain**: Better request batching, reduced latency variance

---

### 4. Fused Sampling + KV Append

**Problem**: Two separate kernels: Sampler (get token) → KVCache_Append (write cache)

**Idea**: Fuse them into one kernel (less syncs, better data reuse)

```
Before (2 kernels):
  GPU: [Sampler] → [KVCache_Append]
       ↓
       Outputs token_id, logits[1, vocab]
       
After (1 fused kernel):
  GPU: [FusedSamplerCacheAppend]
       ├─ On-device softmax + top-k sampling
       ├─ Output: token_id
       └─ Append [1, hidden] to cache
           ↓
       GPU: [lm_head] (compute logits for next token)
            (overlaps with first decode layer)
```

**Implementation**:
```rust
pub unsafe fn role_sampler_cache_append_fused(
    token_logits: *const f32,      // [1, vocab]
    cache: *mut f32,                // [seq_len, hidden]
    cache_pos: u32,
    hidden: *const f32,             // [1, hidden] (last layer output)
    output: *mut i32,               // Output token_id
) {
    // Thread 0: Run softmax + top-k sampling
    let token_id = sample_from_logits(token_logits);
    
    // Threads 1..N: Append hidden to cache (coalesced)
    for i in threadIdx.x..hidden_dim {
        cache[cache_pos * hidden_dim + i] = hidden[i];
    }
    
    __syncthreads();
    
    // Atomically increment cache position
    if threadIdx.x == 0 {
        atomicInc(&cache_pos);
        output[0] = token_id;
    }
}
```

**Expected gain**: Reduce 2 syncs to 1, better cache locality

---

### 5. Speculative Execution (Multi-Token Hypotheses)

**Problem**: Generate one token at a time; GPU has idle capacity

**Idea**: Speculatively compute K hypotheses for next N tokens, verify against true path

```
Decode iteration 1:
  → Token A (certain)
  
Speculative iteration 1 (while real layer computes A→B):
  Hypotheses for B: [B1, B2, B3, B4]
  Embed all 4 in single forward pass (vectorized)
  
Real iteration 2:
  → Token B confirmed
  If B == B_true: accept all 4 hypotheses (draft 4 tokens!)
  If B != B_true: reject, restart with true path
  
Net: 5 token generations in 1.2× the latency of 1
```

**Architecture**:
```rust
pub struct SpeculativeDecodeState {
    pub hypothesis_tokens: Vec<Vec<i32>>,  // [num_hypothesis, depth]
    pub hypothesis_logits: Vec<Vec<f32>>,  // [num_hypothesis][vocab_size]
}

impl HaikuSan {
    pub async fn decode_with_speculation(
        &mut self,
        initial_token: i32,
    ) -> Result<Vec<i32>> {
        let mut tokens = vec![initial_token];
        let mut spec_state = SpeculativeDecodeState::new(num_hypothesis=4);
        
        loop {
            // Generate true next token
            let true_token = self.decode_single(tokens.last().unwrap()).await?;
            tokens.push(true_token);
            
            // Speculatively compute hypotheses for token after next
            let next_hypotheses = self.speculate_next_tokens(
                tokens.last().unwrap(),
                depth=4,  // Hypothesize 4 tokens ahead
            ).await?;
            
            // Next iteration will verify these
            spec_state.hypothesis_tokens = next_hypotheses;
            
            if self.should_stop(&tokens)? {
                break;
            }
        }
        
        Ok(tokens)
    }
}
```

**Expected gain**: 3-4× token generation rate on speculative paths (real-time agent reasoning)

---

## Extension Comparison

| Extension | Latency | Throughput | Complexity | Risk |
|-----------|---------|-----------|-----------|------|
| **Pipeline (L0→L1)** | — | +50% | LOW | LOW |
| **Token streaming** | -20% | +10% | MED | MED |
| **Dynamic batching** | — | +30% | MED | LOW |
| **Fused sampler** | -5% | — | LOW | LOW |
| **Speculative** | +20% latency | **+300%** throughput | HIGH | MED |

---

## Combined Stack: Full Architecture

```
┌──────────────────────────────────────────────────────────┐
│  HAIKU-SAN ORCHESTRATOR (Multi-mode)                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Mode 1: Prefill (batch of requests)                     │
│    - Pipeline layers (L0, L1, L2 concurrent)            │
│    - Dynamic batch sizing (add requests on the fly)      │
│    - Role kernels (RMSNormBatch, GEMVBatchPrefill, ...) │
│                                                          │
│  Mode 2: Decode (streaming generation)                   │
│    - Token streaming (pipelined layers)                  │
│    - Speculative (hypothesize next tokens)              │
│    - Fused sampler+cache (single kernel)                │
│    - Role kernels (RMSNormSingle, GEMVDecodeSingle, ...) │
│                                                          │
│  Mode 3: Verification (check speculative hypotheses)     │
│    - Batch verify guesses against true token            │
│    - Accept/reject hypotheses                            │
│                                                          │
└──────────────────────────────────────────────────────────┘
              ↓
    GPU Kernels (Role-Based)
```

---

## Performance Model: Stacked Extensions

```
Baseline (current per-kernel): 
  Prefill 512 tokens: 50 ms
  Decode 1 token:  6 ms
  Decode N tokens: 6ms + 6ms×(N-1)
  
With pipeline:
  Prefill 512: 35 ms (-30%)
  Decode 1: 6 ms (—)
  
With token streaming:
  Decode 1: 6 ms
  Decode 2-5: ~0.5 ms each (overlapped)  
  
With speculation (4 hypotheses):
  Decode 1: 6 ms
  Decode 2-5 (speculative): 1.5 ms each
  If all correct: 5 tokens in ~10 ms (was ~30 ms)
  
Total 10-token decode:
  Baseline: 6 + 6×9 = 60 ms
  Full stack: 6 + 1.5×3 + 6 + 1.5×5 = 25 ms (2.4× speedup)
```

---

## Implementation Phases

### Phase 1: Role-Based Kernels (Immediate)
- RoleRMSNormSingle, RoleGEMVDecodeSingle, RoleFlashAttnSingle
- RoleKVCache_Append, RoleKVCache_Gather
- RoleSampler_Single
- Expected: +20% decode speedup

### Phase 2: Pipeline Parallelism (2-3 weeks)
- Layer-level pipelining in Haiku-San
- Multi-batch prefill orchestration
- Expected: +50% prefill throughput

### Phase 3: Token Streaming (3-4 weeks)
- Async layer execution (non-blocking per layer)
- Early sampler dispatch
- Expected: -20% decode latency

### Phase 4: Speculative Execution (4-6 weeks)
- Multi-hypothesis attention kernel
- Verification + accept/reject logic
- Expected: +200-300% on speculative paths

### Phase 5: Fusion Optimization (Ongoing)
- Fused sampler+cache_append
- Fused RMSNorm into previous layer output
- Expected: -5% per-op latency

---

## Why This Stack Matters

1. **Orthogonal gains**: Pipeline + streaming + speculation are independent (can combine)
2. **Progressive complexity**: Start with role kernels (simple), add speculation (advanced)
3. **Real-world impact**: Agent inference needs fast speculation (reasoning phase)
4. **GPU efficiency**: Keeps GPU occupied (pipelining + streaming + spec)
5. **Scalability**: Works on different hardware (role kernels tune per-device)

---

## Next Steps

1. **Implement role-based kernels** (Phase 1)
   - Decode roles first (immediate 20% gain)
   - Test on Llama-1B

2. **Add layer pipelining** (Phase 2)
   - Prefill multiple batches concurrently
   - Measure throughput improvement

3. **Token streaming** (Phase 3)
   - Non-blocking per-layer execution
   - Measure latency reduction

4. **Speculative execution** (Phase 4)
   - If speculative ratio > 50%, huge win
   - If <30%, high variance (not useful)

---

## Summary

**Extensions build on Haiku-San:**
- **Role-based kernels**: Optimize for phase (prefill vs decode)
- **Pipeline parallelism**: Layers execute concurrently
- **Token streaming**: Early sampler dispatch
- **Speculative execution**: Multi-token hypotheses (3-4× on lucky paths)
- **Fusion optimization**: Reduce inter-kernel overhead

**Expected combined speedup**: 2-3× latency, 3-4× throughput

This is the **long-term trajectory** for hybrid kernels: from safe + correct → fast + speculative → agent-native inference.
