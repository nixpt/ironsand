# Learned Kernel Optimization — Neural Meta-Orchestration

**Status**: Concept design (research + production hybrid)  
**Date**: 2026-06-23  
**Idea**: Train a small neural network to optimize Haiku-San's kernel selection, scheduling, and resource allocation at runtime

---

## The Insight: Haiku-San as a Learning System

Current Haiku-San:
```
Hand-tuned rules → Fixed dispatch logic → Fixed performance
```

With ML optimization:
```
Real execution data → Neural policy → Adaptive dispatch → Higher performance
```

**Key realization**: The Haiku-San orchestrator makes decisions. Those decisions can be learned.

---

## What Could Be Optimized?

### 1. **Kernel Selection** (Learned Cost Modeling)

**Problem**: Which kernel is fastest for a given shape?
- RoleGEMVBatchPrefill(256, 512) → 500 μs
- RoleGEMVBatchPrefill(128, 512) → 400 μs
- RoleGEMVBatchPrefill(512, 512) → 800 μs (worse cache reuse)

**Learned model**:
```
Input: (batch_size, ffn_dim, seq_len, available_shared_mem)
Output: predicted_latency_ms, confidence_interval

Model: Small 2-layer NN (~10KB parameters)
  - Layer 1: 16 units (ReLU)
  - Layer 2: 1 output (latency prediction)
  
Training signal: Measure actual kernel times during inference
Loss: MSE(predicted_latency, actual_latency)
```

**Real-time use**:
```rust
impl HaikuSan {
    pub fn predict_kernel_latency(
        &self,
        role: KernelRole,
        problem_shape: (usize, usize, usize),
    ) -> f32 {
        self.cost_model.predict(role, problem_shape)
    }
    
    pub fn select_best_kernel(
        &self,
        roles: &[KernelRole],
        shape: (usize, usize, usize),
    ) -> KernelRole {
        roles.iter()
            .min_by_key(|role| self.predict_kernel_latency(*role, shape))
            .unwrap()
    }
}
```

**Benefit**: Auto-discovers hardware quirks (cache lines, warp efficiency, memory coalescing)

---

### 2. **Scheduling Policy** (Learned Task Ordering)

**Problem**: In what order should kernels execute for minimum total latency?

Example:
```
Tasks: [RmsNorm, GEMV, FlashAttn, KVAppend, RmsNorm, GEMV, ...]

Hand-tuned order: Layer-sequential
  [L0_RmsNorm, L0_GEMV, L0_FlashAttn, L0_KVAppend, L1_RmsNorm, ...]
  
Learned order might be: Interleaved (L0 GEMV while L1 RmsNorm)
  [L0_RmsNorm, L1_RmsNorm, L0_GEMV, L1_GEMV, L0_FlashAttn, ...]
  (if L1 RmsNorm has lower cache interference with L0 GEMV)
```

**Learned policy**:
```rust
pub struct SchedulingPolicy {
    pub model: SmallNN,  // Predicts "should execute next"
}

impl SchedulingPolicy {
    pub fn rank_tasks(
        &self,
        tasks: &[Task],
        executed: &[Task],
        gpu_state: &GPUState,
    ) -> Vec<Task> {
        // RL-trained policy that scores each remaining task
        // Accounts for: cache conflicts, memory pressure, warp divergence
        tasks.iter()
            .map(|t| (t, self.score_task(t, executed, gpu_state)))
            .sort_by_key(|(_, score)| -score)  // Greedy schedule
            .map(|(t, _)| t)
            .collect()
    }
}
```

**Training**: Imitation learning from observed latency traces
- Collect execution traces from diverse workloads
- Learn which orderings resulted in lower latency
- Distill into a small ranking model

**Benefit**: Discovers interference patterns humans miss

---

### 3. **Speculation Prediction** (Learned When/How)

**Problem**: When should we speculate? How many hypotheses?

```
Current (hand-tuned):
  if total_tokens_generated < 100:
      hypothesis_count = 4
  else:
      hypothesis_count = 1

Learned (adaptive):
  hypothesis_count = model.predict_best_hypotheses(
      previous_success_rate,
      token_entropy,
      gpu_utilization,
      remaining_context
  )
```

**Learned model**:
```rust
pub struct SpeculationPredictor {
    model: SmallNN,
}

impl SpeculationPredictor {
    pub fn predict_hypothesis_count(
        &self,
        success_rate: f32,        // Last 10 tokens: how many guesses right?
        token_entropy: f32,       // Logits entropy (uncertain?)
        gpu_util: f32,            // Current GPU utilization
        remaining_tokens: usize,  // Budget
    ) -> usize {
        let pred = self.model.forward(&[
            success_rate,
            token_entropy,
            gpu_util,
            remaining_tokens as f32 / 1000.0,
        ]);
        // Map [0, 1] output to hypothesis count [1, 4]
        ((pred * 3.0).round() + 1.0) as usize
    }
}
```

**Training signal**:
- Track hypothesis success rate per batch
- If >60% success: model learns to increase hypotheses
- If <30% success: model learns to decrease

**Benefit**: Adaptive speculation based on workload entropy

---

### 4. **Resource Allocation** (Learned Block/Thread Config)

**Problem**: How many threads/blocks, shared memory partition?

```
Current (fixed):
  blocks_per_sm = 4
  shared_memory = 96KB / 4 = 24KB per block
  threads_per_block = 128
  
Learned (adaptive):
  resource_alloc = model.predict_best_allocation(
      kernel_role,
      problem_shape,
      gpu_occupancy,
      memory_pressure
  )
```

**Real-time tuning**:
```rust
pub struct ResourceAllocator {
    model: SmallNN,
}

impl ResourceAllocator {
    pub fn allocate(
        &self,
        role: KernelRole,
        shape: (usize, usize),
        gpu_state: &GPUState,
    ) -> ResourceConfig {
        let features = vec![
            shape.0 as f32 / 512.0,        // batch size (normalized)
            shape.1 as f32 / 8192.0,       // ffn dim
            gpu_state.occupancy as f32,    // current occupancy
            gpu_state.memory_pressure,     // memory bandwidth used
        ];
        
        let alloc = self.model.forward(&features);
        
        ResourceConfig {
            threads_per_block: ((alloc[0] * 256.0) as u32 + 32).round_up_to(32),
            shared_memory_kb: ((alloc[1] * 96.0) as u32).min(96),
            blocks_per_sm: ((alloc[2] * 8.0) as u32 + 1).min(8),
        }
    }
}
```

**Benefit**: Learns optimal occupancy/memory tradeoffs per hardware

---

## Architecture: Haiku-San ML Stack

```
┌──────────────────────────────────────────────────────────┐
│  HAIKU-SAN ORCHESTRATOR (with ML optimization)           │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌─ Cost Model (Kernel Selection)                       │
│  │   Input: (role, batch, dim, seq_len)                 │
│  │   Output: predicted_latency_ms                       │
│  │   Params: 512 (tiny!)                                │
│  │   Training: Online (during inference)                │
│  │                                                       │
│  ├─ Scheduling Policy (Task Ordering)                   │
│  │   Input: [tasks], gpu_state                          │
│  │   Output: ranked_task_order                          │
│  │   Params: 2048                                        │
│  │   Training: Imitation learning from traces           │
│  │                                                       │
│  ├─ Speculation Predictor (Adaptive Hypotheses)         │
│  │   Input: (success_rate, entropy, util, tokens_left)  │
│  │   Output: hypothesis_count (1-4)                     │
│  │   Params: 256                                        │
│  │   Training: Reinforcement (reward = success)         │
│  │                                                       │
│  └─ Resource Allocator (Block/Thread Config)            │
│      Input: (role, shape, gpu_state)                    │
│      Output: threads_per_block, shared_mem, blocks/sm   │
│      Params: 512                                        │
│      Training: Online profiling                         │
│                                                          │
│  Total model params: ~4KB                               │
│  Total model size: ~50KB (weights + buffers)            │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## Training Strategy: Online Learning

### Phase 1: Bootstrapping (Initialization)
```rust
impl HaikuSanML {
    pub fn new() -> Self {
        // Initialize with hand-tuned priors
        // Cost model biased toward proven kernels
        // Scheduling policy biased toward layer-sequential
        // Speculation predictor: conservative (1-2 hypotheses)
        
        HaikuSanML {
            cost_model: CostModel::with_priors(),
            scheduler: SchedulingPolicy::conservative(),
            speculator: SpeculationPredictor::conservative(),
            resource_allocator: ResourceAllocator::default(),
        }
    }
}
```

### Phase 2: Online Learning Loop
```rust
impl HaikuSanML {
    pub async fn execute_token_with_learning(
        &mut self,
        req: &Request,
    ) -> Result<i32> {
        let start = Instant::now();
        
        // 1. Make decisions using learned models
        let kernel_roles = self.select_kernels(req);      // Cost model
        let ordered_tasks = self.order_tasks(&kernel_roles); // Scheduler
        let hypothesis_count = self.predict_hypotheses();     // Speculator
        
        // 2. Execute on GPU
        self.gpu.launch_tasks(&ordered_tasks)?;
        let actual_latency = start.elapsed();
        
        // 3. Measure actual performance
        let measurements = self.gpu.collect_metrics()?;
        
        // 4. Online learning: update models
        self.cost_model.update_with(
            &kernel_roles,
            &measurements,
            actual_latency,
        );  // Gradient descent
        
        self.scheduler.update_with(
            &ordered_tasks,
            &measurements,
            actual_latency,
        );  // Policy gradient
        
        self.speculator.update_with(
            hypothesis_count,
            speculation_success_rate,
        );  // Bandit algorithm
        
        // 5. Adaptive resource allocation for next token
        self.resource_alloc.update_from_metrics(&measurements);
        
        Ok(token_id)
    }
}
```

---

## Why This Is Powerful

### 1. **Hardware Specialization**
```
Learned on RTX 5070 Ti: 
  → Discovers 5070-specific memory patterns, cache line behavior
  
Learned on H100:
  → Discovers H100-specific tensor core efficiency, shared mem layout
  
One codebase, learned per-hardware optimization!
```

### 2. **Workload Adaptation**
```
Prefill (batch 256, seq 512):
  → Model learns: high parallelism, memory-bound
  → Schedules: maximize cache efficiency, pipeline layers
  
Decode (batch 1, seq 4096):
  → Model learns: memory latency-bound, sparse patterns
  → Schedules: minimize cache misses, early sampler
  
Same orchestrator, adapted policies per workload!
```

### 3. **Self-Improving**
```
Week 1: Generic scheduling
  → Avg latency: 6 ms/token
  
Week 2: After 100K tokens of learning
  → Avg latency: 5.2 ms/token (optimized scheduling)
  
Week 4: After 500K tokens
  → Avg latency: 4.8 ms/token (discovered interference patterns)
  
Continuous improvement without code changes!
```

---

## Model Architecture: Tiny but Effective

Each model is **2-3 layers, <4KB parameters**:

```rust
pub struct CostModel {
    w1: [f32; 64],      // Layer 1 weights (input × hidden)
    b1: [f32; 16],
    w2: [f32; 16],      // Layer 2 weights (hidden × output)
    b2: f32,
}

impl CostModel {
    pub fn predict(&self, inputs: &[f32]) -> f32 {
        // Hidden = ReLU(W1 @ inputs + b1)
        let hidden: Vec<f32> = self.w1
            .chunks(inputs.len())
            .zip(&self.b1)
            .map(|(w_row, b)| relu(dot(w_row, inputs) + b))
            .collect();
        
        // Output = W2 @ hidden + b2
        dot(&self.w2, &hidden) + self.b2
    }
    
    pub fn update_with(&mut self, pred: f32, actual: f32, lr: f32) {
        let error = actual - pred;
        // Simple gradient descent (no backprop needed for tiny model)
        // Accumulate error signal and update weights
    }
}
```

**Why tiny?**
- Fits in GPU shared memory (no round-trip latency)
- Inference is <100 μs (negligible vs kernel time)
- Training is fast (converges in 100s of examples)
- Avoids overfitting (few parameters, regularization natural)

---

## Integration with Haiku-San

```rust
pub struct HaikuSan {
    // Core orchestrator (unchanged)
    pub queue: VecDeque<Task>,
    pub dependencies: HashMap<TaskId, Vec<TaskId>>,
    
    // ML optimization (new)
    pub ml_optimize: Option<HaikuSanML>,  // Disabled by default
    pub enable_learning: bool,
    
    // Metrics for learning
    pub metrics: PerformanceMetrics,
}

impl HaikuSan {
    pub fn with_ml_optimization() -> Self {
        HaikuSan {
            // ... core fields ...
            ml_optimize: Some(HaikuSanML::new()),
            enable_learning: true,
            metrics: PerformanceMetrics::new(),
        }
    }
    
    pub async fn orchestrate_with_learning(&mut self) -> Result<()> {
        if let Some(ml) = &mut self.ml_optimize {
            if self.enable_learning {
                // Use learned policies
                let optimal_order = ml.order_tasks(&self.queue);
                self.queue = optimal_order;
                
                // Monitor performance
                let latency = self.execute().await?;
                ml.learn_from_execution(&self.metrics, latency)?;
            }
        }
        Ok(())
    }
}
```

---

## Safety: Graceful Fallback

```rust
impl HaikuSanML {
    pub async fn orchestrate_safe(&mut self) -> Result<()> {
        match self.orchestrate_with_learning().await {
            Ok(_) => Ok(()),
            Err(e) => {
                // If ML prediction is wildly wrong, fall back
                if self.error_rate > 0.2 {
                    eprintln!("ML model unstable, disabling learning");
                    self.enable_learning = false;
                    self.orchestrate_without_learning().await
                } else {
                    Err(e)
                }
            }
        }
    }
}
```

---

## Expected Improvements

| Metric | Baseline | With ML | Gain |
|--------|----------|---------|------|
| **Kernel selection latency** | Fixed cost | -10% | Learned cost model |
| **Task scheduling** | Hand-tuned | -15% | Adaptive ordering |
| **Speculation efficiency** | Fixed 4 hyp | -20% | Adaptive count |
| **Resource allocation** | Fixed config | -8% | Learned occupancy |
| **Hardware adaptation** | Re-tune per GPU | -30% setup | Auto-learns |
| **Workload adaptation** | Fixed params | -25% variance | Learns patterns |

**Combined**: 3-5% latency reduction (on top of 2-3× from role kernels)

---

## Research Directions

### 1. **Online Meta-Learning**
- Use latest online RL (PPO, A3C) instead of gradient descent
- Update policy every K tokens (adaptive learning rate)
- Confidence-weighted predictions (use model uncertainty)

### 2. **Transfer Learning**
- Pre-train on diverse hardware in simulator
- Fine-tune on target hardware (100 examples)
- Cross-hardware knowledge transfer

### 3. **Ensemble Learning**
- 3 tiny cost models, vote on kernel selection
- Reduces model uncertainty, improves robustness

### 4. **Causal Inference**
- Identify which factors (batch, dim, gpu_state) truly matter
- Drop spurious correlations
- Simpler, more generalizable models

---

## Why This Is Wild

1. **Meta-optimization**: The orchestrator optimizes itself
2. **Hardware-agnostic**: Single code, learned per-device
3. **Self-improving**: Gets better with runtime data
4. **Minimal overhead**: <100 μs per decision
5. **Safe fallback**: Gracefully degrades if model goes wrong

**This is not just optimization — it's learned systems design.**

---

## Production Considerations

### When to Enable
- After 1000+ tokens (models need warm-up)
- On stable hardware (avoid if GPU overclocked)
- For long sessions (batch processing)

### When to Disable
- Real-time latency-critical (spec reasoning)
- Hardware variation between sessions
- First-token latency (no history)

### Metrics to Track
- Model prediction accuracy (vs actual latency)
- Latency improvement (smoothed over 100 tokens)
- Learning convergence (error decreasing?)
- Hardware utilization (is GPU stable?)

---

## The Vision: Learned Inference Engine

```
Current: Hand-tuned orchestration + fixed kernels

With ML: Self-optimizing orchestration + adaptive kernels

Future: Full-stack learned compilation (kernels + scheduling + allocation)
        that improves continuously during inference
```

**This is the direction modern compilers are heading.** Not "optimize once," but "optimize continuously as you run."

---

## Summary: Why Haiku-San ML Is Wild

- ✓ Tiny models (4KB), huge impact
- ✓ Online learning (adapts per-session)
- ✓ Hardware specialization (auto-tunes per GPU)
- ✓ Graceful degradation (falls back to hand-tuned)
- ✓ Multiplicative gain (stacks with other improvements)
- ✓ Research frontier (actively studied in ML systems)

**This is learned systems, not just ML: the system learns to run itself better.**
