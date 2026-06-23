# 4-Week Implementation Plan — Ship Role-Based Decode Kernels to Zorro

**Goal**: Get 20% decode speedup into zorro by integrating role-based kernels for the decode path  
**Timeline**: 4 weeks (28 days)  
**Focus**: Decode optimization (single-token path), NOT prefill (deferred to phase 2)  
**Success Metric**: Measure 20%+ latency reduction in single-token generation

---

## Week 1: Implement Decode Roles (3 kernels)

### Days 1-3: RoleRMSNormSingle
**Task**: Implement lightweight single-row normalization kernel
```rust
// src/kernels/roles/rms_norm_single.rs
pub unsafe fn role_rms_norm_single(
    input: *const f32,     // [1, hidden_dim]
    output: *mut f32,      // [1, hidden_dim]
    hidden_dim: usize,
    eps: f32,
) {
    // Compute variance (single thread is enough)
    // Normalize: output = input / sqrt(var + eps)
    // Target: <100 μs per call
}
```
**Acceptance**: Latency <100 μs, L2-rel <1e-4 vs CPU

### Days 4-5: RoleGEMVDecodeSingle
**Task**: Thin GEMV kernel optimized for 1×hidden matrix-vector
```rust
// src/kernels/roles/gemv_decode_single.rs
pub unsafe fn role_gemv_decode_single(
    matrix: *const f32,    // [hidden_dim, hidden_dim]
    vector: *const f32,    // [hidden_dim]
    output: *mut f32,      // [hidden_dim]
) {
    // Single-row GEMV: each thread handles 1 output element
    // Coalesce across hidden_dim (cache-friendly)
    // Target: <500 μs per layer (8192→8192)
}
```
**Acceptance**: Latency <500 μs/layer, correctness verified

### Days 6-7: RoleFlashAttnSingle
**Task**: Single-token attention using cached K/V
```rust
// src/kernels/roles/flash_attn_single.rs
pub unsafe fn role_flash_attn_single(
    q: *const f32,         // [1, num_heads, head_dim]
    k_cache: *const f32,   // [seq_len, num_heads, head_dim]
    v_cache: *const f32,   // [seq_len, num_heads, head_dim]
    output: *mut f32,      // [1, num_heads, head_dim]
    seq_len: usize,
) {
    // Online softmax over sequence (single query)
    // Reuse cached K/V (no computation)
    // Target: <2000 μs per layer (1024-token cache)
}
```
**Acceptance**: Latency <2000 μs/layer, supports caching

---

## Week 2: Integrate Roles into Haiku-San

### Days 8-10: Phase-Aware Dispatch
**Task**: Wire role selection into Haiku-San orchestrator
```rust
// src/haiku_san.rs
impl HaikuSan {
    pub fn decode_phase(&mut self) -> KernelRole {
        match self.current_phase {
            Phase::Decode => {
                // Route to role kernels
                vec![
                    KernelRole::RMSNormSingle,
                    KernelRole::GEMVDecodeSingle,
                    KernelRole::FlashAttnSingle,
                    KernelRole::KVCacheAppend,
                ]
            }
            _ => panic!("Wrong phase"),
        }
    }
}
```
**Acceptance**: Kernels dispatched correctly, no crashes

### Days 11-14: KV Cache Management
**Task**: Implement streaming KV cache for decode
```rust
// src/kv_cache.rs
pub struct KVCache {
    k: DeviceBuffer<f32>,  // [max_seq, num_heads, head_dim]
    v: DeviceBuffer<f32>,
    position: u32,         // Current write position
}

impl KVCache {
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32]) {
        // Write new K/V at position
        // Increment position
    }
    
    pub fn get_for_attention(&self) -> (&[f32], &[f32]) {
        // Return full cache for single-token attention
    }
}
```
**Acceptance**: Cache correctly tracks position, no overwrites

---

## Week 3: Integration with Zorro

### Days 15-18: Wire into Decode Loop
**Task**: Replace current per-kernel launches with Haiku-San orchestration
```rust
// zorro/src/cuda/decode.rs (BEFORE)
for layer in &model.layers {
    launch!(rmsnorm_kernel, ...)?;
    launch!(gemv_kernel, ...)?;
    launch!(attention_kernel, ...)?;
    // ~3 syncs per layer
}

// zorro/src/cuda/decode.rs (AFTER)
let mut orchestrator = HaikuSan::decode_phase();
for layer in &model.layers {
    orchestrator.submit_role(RoleRMSNormSingle, ...);
    orchestrator.submit_role(RoleGEMVDecodeSingle, ...);
    orchestrator.submit_role(RoleFlashAttnSingle, ...);
}
orchestrator.launch_all_async();
// 1 sync total
```
**Acceptance**: Code compiles, passes existing correctness tests

### Days 19-21: Validation
**Task**: Run existing zorro test suite, verify correctness
```
[ ] Paris oracle test suite passes
[ ] Quantization (Q4K) integration works
[ ] Multi-turn KV caching works
[ ] Sampler produces same tokens
```
**Acceptance**: All tests pass, no regressions

---

## Week 4: Measurement & Shipping

### Days 22-24: Benchmarking
**Task**: Measure decode latency before/after role kernels
```rust
// benches/decode_latency.rs
fn bench_decode_baseline() {
    // Current per-kernel approach
    // Measure: 1000 tokens, median latency
}

fn bench_decode_with_roles() {
    // New Haiku-San + roles
    // Measure: 1000 tokens, median latency
}

// Expected: 6 ms → 5 ms (20% gain from reduced syncs + optimization)
```
**Acceptance**: 
- Baseline: 6-8 ms/token
- With roles: 5-6.5 ms/token
- Speedup: ≥15%

### Days 25-27: Documentation
**Task**: Document the integration for zorro maintainers
```
- README: How to use Haiku-San decode
- Architecture: Why role kernels are faster
- Configuration: How to tune per-GPU
- Migration: How to opt-in/out
```
**Acceptance**: Docs clear enough for colleague to extend

### Day 28: Ship
**Task**: Create PR, merge, tag release
```bash
git tag v1.0-decode-roles
# Deploy to zorro
```
**Acceptance**: 20% decode speedup in production

---

## Daily Checklist Template

```
## Week 1, Day 1: Start RoleRMSNormSingle

- [ ] Skeleton kernel compiles
- [ ] Inline asm works (bar.sync, shfl)
- [ ] CPU reference implementation matches
- [ ] Latency <100 μs (time with events)
- [ ] Commit with message "Implement RoleRMSNormSingle"

## Standup Note
Status: BLOCKED/IN_PROGRESS/DONE
Time spent: ___ hours
Blockers: ___
Next: ___
```

---

## Risk Mitigation

### Risk: Role kernels have wrong latency assumptions
**Mitigation**: Day 7 → measure actual latencies on 5070 Ti before integrating
**Gate**: If any role exceeds target by >20%, pivot approach

### Risk: Integration with zorro breaks existing tests
**Mitigation**: Run full test suite daily (Days 15+)
**Gate**: Don't advance to measurement if any test fails

### Risk: Speedup doesn't materialize (syncs not the bottleneck)
**Mitigation**: Day 22 → detailed profiling (GPU events, SM utilization)
**Gate**: If speedup <10%, investigate why before shipping

### Fallback: If days fall behind
- Defer KV cache optimization (Day 11-14) → use simple global cache
- Defer benchmarking (Days 22-24) → ship with one test config
- Defer detailed docs → minimal README, plan follow-up

---

## Success Criteria (Ship Gate)

✅ **All three role kernels working**
- RoleRMSNormSingle: <100 μs
- RoleGEMVDecodeSingle: <500 μs
- RoleFlashAttnSingle: <2000 μs (1024-token cache)

✅ **Integrated into zorro**
- Decode loop uses Haiku-San
- All existing tests pass
- No regressions

✅ **Measured 15-20% speedup**
- Latency: 6 ms → 5 ms (or better)
- Consistent across 1000-token runs
- Margin of error: <10%

✅ **Documented for handoff**
- Integration guide written
- Configuration guide written
- Fallback/debugging guide written

---

## Post-Ship (Phase 2, deferred)

Once role kernels are shipping:
- **Week 5-6**: Add prefill roles (RMSNormBatch, GEMVBatchPrefill)
- **Week 7-8**: Pipeline parallelism
- **Week 9-10**: Token streaming
- Future: Speculation, learning optimization

---

## Weekly Sync Template

```
## Week N Standup
Completed: [list of milestones]
On track: YES / BLOCKED
Speedup measured: X% (target 15-20%)
Risks: [any new blockers?]
Next week: [next phase]
```

---

## Commits Expected This Month

```
Week 1:
  - impl: RoleRMSNormSingle kernel
  - impl: RoleGEMVDecodeSingle kernel
  - impl: RoleFlashAttnSingle kernel

Week 2:
  - feat: Phase-aware kernel dispatch
  - feat: KV cache streaming
  - test: Role kernel correctness

Week 3:
  - feat: Integrate role kernels into zorro decode loop
  - test: Verify no regressions (Paris oracle)
  - docs: Integration guide

Week 4:
  - perf: Benchmark decode latency (baseline vs roles)
  - docs: Configuration + troubleshooting
  - release: v1.0-decode-roles

Total: 12 commits, all reviewed, all tested
```

---

## Definition of Done

A role kernel is **DONE** when:
1. ✅ Kernel compiles to PTX (no warnings)
2. ✅ Correctness verified (L2-rel <1e-4 vs CPU)
3. ✅ Latency measured (within target)
4. ✅ Integrated into Haiku-San (dispatched correctly)
5. ✅ Zorro tests pass (no regressions)
6. ✅ Benchmarked (speedup measured)
7. ✅ Documented (how to use, when to use)
8. ✅ Committed (one logical commit per kernel)

---

## Key Insights for Fast Shipping

1. **Decode only (not prefill)**: Single-token path is simpler, less state
2. **Reuse caches**: KV is already cached, just index it
3. **Lightweight roles**: RMSNormSingle is <100 lines, GEMVDecodeSingle <200
4. **No speculation**: Deferring means simpler logic, faster iteration
5. **One metric**: Latency is the goal; don't measure TFLOPs or power
6. **Daily commits**: Small batches, less risky, easier to debug

---

## This Month's Goal

**Zorro users can run decode 20% faster by opting into role-based kernels.**

Not "design the full system" — **ship a working component, measure it, repeat.**

This is how you build confidence in the architecture: prove each piece works before stacking them.
