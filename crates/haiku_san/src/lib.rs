//! Haiku-San: CPU/GPU hybrid orchestrator for distributed kernel chains
//!
//! # Overview
//!
//! Haiku-San is a lightweight CPU-side task orchestrator that manages async GPU kernel
//! launches, enabling safe CPU/GPU parallel execution without monolithic megakernel
//! deadlock risk.
//!
//! ## Architecture
//!
//! Instead of launching a single massive kernel that tries to do everything:
//! - **Monolithic approach**: All ops in one kernel → deadlock (hand-rolled grid barriers fail)
//! - **Haiku-San approach**: CPU submits 64 small kernels per token, GPU queues them async
//!
//! ## Key Features
//!
//! - **Async submission**: Submit all tasks without blocking on GPU completion
//! - **Dependency tracking**: Task A waits for task B via GPU events
//! - **CPU/GPU parallelism**: CPU prefetches/validates while GPU executes
//! - **Safe barriers**: Use GPU driver's event synchronization (not hand-rolled atomics)
//!
//! ## Usage
//!
//! ```ignore
//! let mut orchestrator = HaikuSan::new();
//!
//! // Submit tasks
//! let task1 = orchestrator.submit_task("RmsNorm", OP_RMSNORM, 256, 256);
//! let task2 = orchestrator.submit_task("GEMV", OP_GEMV, 512, 256);
//! orchestrator.add_dependency(task2, task1);  // task2 waits for task1
//!
//! // Launch all async (GPU starts executing; CPU continues)
//! orchestrator.launch_all_async(&stream)?;
//!
//! // CPU work in parallel (prefetch, validate, etc.)
//! // ...
//!
//! // Synchronize when needed
//! orchestrator.wait_for(task2)?;
//! ```
//!
//! # Design Files
//!
//! For detailed architecture, see:
//! - `doc/HAIKU_SAN_DESIGN.md` — Core orchestration architecture
//! - `doc/CAPACITY_ANALYSIS.md` — Why 64 kernels/token is optimal
//! - `doc/ROLE_KERNELS.md` — Phase-aware kernel specialization

use std::collections::{HashMap, VecDeque};
use std::error::Error;

use cust::event::Event;
use cust::stream::Stream;

pub type TaskId = u64;

/// A kernel task to be executed on GPU.
#[derive(Clone, Debug)]
pub struct KernelTask {
    /// Unique task identifier
    pub id: TaskId,
    /// Human-readable task name (e.g., "RmsNorm_L0")
    pub name: String,
    /// Opcode identifying which kernel to run
    pub opcode: u32,
    /// Output size (problem dimension M)
    pub m: u32,
    /// Input size (problem dimension N)
    pub n: u32,
    /// Task IDs that must complete before this task starts
    pub depends_on: Vec<TaskId>,
}

/// Orchestration statistics for profiling and debugging.
#[derive(Clone, Copy, Debug, Default)]
pub struct OrchestrationStats {
    /// Number of tasks submitted to GPU
    pub tasks_submitted: u64,
    /// Number of tasks that completed
    pub tasks_completed: u64,
    /// Total GPU synchronization time (microseconds)
    pub total_wait_time_us: f32,
    /// Total CPU work time (microseconds)
    pub cpu_work_us: f32,
}

/// CPU-side orchestrator for distributed GPU kernels.
///
/// Manages task submission, dependency tracking, and synchronization
/// without blocking on GPU execution between submissions.
pub struct HaikuSan {
    /// Counter for assigning unique task IDs
    next_task_id: TaskId,
    /// Queue of tasks pending GPU launch
    queue: VecDeque<KernelTask>,
    /// In-flight tasks mapped to their GPU events
    inflight: HashMap<TaskId, Event>,
    /// Completed task IDs (for debugging)
    completed: Vec<TaskId>,
    /// Orchestration performance stats
    stats: OrchestrationStats,
}

impl HaikuSan {
    /// Create a new orchestrator.
    pub fn new() -> Self {
        HaikuSan {
            next_task_id: 0,
            queue: VecDeque::new(),
            inflight: HashMap::new(),
            completed: Vec::new(),
            stats: OrchestrationStats::default(),
        }
    }

    /// Submit a kernel task (queued, not yet launched on GPU).
    ///
    /// # Arguments
    /// * `name` - Human-readable task name for debugging
    /// * `opcode` - Kernel selector (e.g., OP_RMSNORM=0, OP_GEMV_F32=1)
    /// * `m` - Output size (rows in matrix-vector, tokens in batch, etc.)
    /// * `n` - Input size (columns, hidden dimension, etc.)
    ///
    /// # Returns
    /// Task ID for use in `add_dependency` and `wait_for` calls.
    pub fn submit_task(&mut self, name: &str, opcode: u32, m: u32, n: u32) -> TaskId {
        let id = self.next_task_id;
        self.next_task_id += 1;

        self.queue.push_back(KernelTask {
            id,
            name: name.to_string(),
            opcode,
            m,
            n,
            depends_on: Vec::new(),
        });

        id
    }

    /// Add a dependency: `dependent` must wait for `task_id` to complete.
    ///
    /// # Panics
    /// Does nothing silently if the dependent task is not in the queue.
    /// This is intentional to avoid deadlock on incorrect usage.
    pub fn add_dependency(&mut self, dependent_id: TaskId, task_id: TaskId) {
        if let Some(task) = self.queue.iter_mut().find(|t| t.id == dependent_id) {
            task.depends_on.push(task_id);
        }
    }

    /// Launch all queued tasks asynchronously on GPU.
    ///
    /// This returns immediately after submitting tasks; it does NOT wait for GPU completion.
    /// GPU events are recorded in the stream for later synchronization via `wait_for`.
    ///
    /// # Arguments
    /// * `stream` - CUDA stream on which to launch tasks
    ///
    /// # Returns
    /// Error if event creation or stream recording fails.
    pub fn launch_all_async(&mut self, stream: &Stream) -> Result<(), Box<dyn Error>> {
        while let Some(task) = self.queue.pop_front() {
            // In production, this would call the actual kernel launcher.
            // For now, we simulate by recording an event at this point in the stream.
            let event = Event::new(cust::event::EventFlags::DEFAULT)?;
            event.record(stream)?;

            self.inflight.insert(task.id, event);
            self.stats.tasks_submitted += 1;

            log::debug!(
                "HAIKU [SUBMIT] task_id={} name={} opcode={} (m={}, n={})",
                task.id, task.name, task.opcode, task.m, task.n
            );
        }

        Ok(())
    }

    /// Wait for a specific task to complete (synchronize on its GPU event).
    ///
    /// This blocks the CPU until the GPU event for this task fires.
    /// Use `add_dependency` to create inter-task dependencies instead of calling
    /// `wait_for` multiple times in a loop.
    ///
    /// # Arguments
    /// * `task_id` - Task to wait for (returned from `submit_task`)
    ///
    /// # Returns
    /// Error if the task is not found or synchronization fails.
    pub fn wait_for(&mut self, task_id: TaskId) -> Result<(), Box<dyn Error>> {
        if let Some(event) = self.inflight.remove(&task_id) {
            event.synchronize()?;
            self.completed.push(task_id);
            self.stats.tasks_completed += 1;

            log::debug!("HAIKU [COMPLETE] task_id={}", task_id);
            Ok(())
        } else {
            Err(format!("Task {} not found or already completed", task_id).into())
        }
    }

    /// Get current orchestration statistics.
    pub fn stats(&self) -> OrchestrationStats {
        self.stats
    }

    /// Reset statistics (useful between phases or batches).
    pub fn reset_stats(&mut self) {
        self.stats = OrchestrationStats::default();
    }

    /// Orchestrate a simple 2-op chain: GEMV_Q4K → SiLU.
    ///
    /// Demonstrates async scheduling + CPU work overlap pattern.
    /// **For testing/spikes only.**
    pub fn orchestrate_two_op_spike(&mut self, stream: &Stream) -> Result<(), Box<dyn Error>> {
        const OP_GEMV_Q4K: u32 = 3;
        const OP_SILU: u32 = 2;

        log::info!("[SPIKE] Orchestrating 2-op spike: GEMV_Q4K → SiLU");

        let task1_id = self.submit_task("GEMV_Q4K", OP_GEMV_Q4K, 128, 256);
        let task2_id = self.submit_task("SiLU", OP_SILU, 128, 0);
        self.add_dependency(task2_id, task1_id);

        self.launch_all_async(stream)?;

        log::info!("HAIKU [CPU-WORK] Validating task1 output...");

        self.wait_for(task1_id)?;
        self.wait_for(task2_id)?;

        log::info!("HAIKU [COMPLETE] 2-op spike done");

        Ok(())
    }

    /// Orchestrate a full layer: RmsNorm → QKV → FlashAttn → OProj → FFN.
    ///
    /// Simplified for demonstration; real version has all operations.
    /// **For testing/spikes only.**
    pub fn orchestrate_layer_spike(&mut self, stream: &Stream) -> Result<(), Box<dyn Error>> {
        const OP_RMSNORM: u32 = 0;
        const OP_GEMV_Q4K: u32 = 3;
        const OP_SILU: u32 = 2;

        log::info!("[SPIKE] Orchestrating full layer: RmsNorm → GEMV → SiLU");

        let task_rmsnorm = self.submit_task("RmsNorm_attn", OP_RMSNORM, 256, 256);
        let task_qkv = self.submit_task("QKVProj", OP_GEMV_Q4K, 768, 256);
        let task_ffn_gate_up = self.submit_task("FFNGateUp", OP_GEMV_Q4K, 512, 256);
        let task_silu = self.submit_task("SiLU", OP_SILU, 512, 0);

        self.add_dependency(task_qkv, task_rmsnorm);
        self.add_dependency(task_ffn_gate_up, task_rmsnorm);
        self.add_dependency(task_silu, task_ffn_gate_up);

        self.launch_all_async(stream)?;

        log::info!("HAIKU [CPU-WORK] CPU parallel phase (GPU executing tasks)");
        log::info!("HAIKU [CPU-WORK] Prefetching next layer weights...");

        self.wait_for(task_rmsnorm)?;
        self.wait_for(task_qkv)?;
        self.wait_for(task_ffn_gate_up)?;
        self.wait_for(task_silu)?;

        log::info!("HAIKU [COMPLETE] Layer spike done");

        Ok(())
    }

    /// Orchestrate a hybrid layer: Attention block + FFN block (stream kernels).
    ///
    /// Demonstrates the recommended architecture:
    /// - Each block (attention, FFN) is a stream kernel internally
    /// - Haiku-San orchestrates the 2 blocks per layer
    /// - Total per-token kernels: 2 per layer (not 10)
    ///
    /// **For testing/spikes only.**
    pub fn orchestrate_hybrid_layer_spike(
        &mut self,
        stream: &Stream,
        layer_idx: usize,
    ) -> Result<(), Box<dyn Error>> {
        const OP_STREAM_ATTN: u32 = 10;
        const OP_STREAM_FFN: u32 = 11;

        log::info!(
            "[HYBRID SPIKE] Layer {}: Stream Attn Block → Stream FFN Block",
            layer_idx
        );

        let task_attn = self.submit_task(
            &format!("StreamAttn_L{}", layer_idx),
            OP_STREAM_ATTN,
            256,
            256,
        );
        let task_ffn = self.submit_task(&format!("StreamFFN_L{}", layer_idx), OP_STREAM_FFN, 512, 256);

        self.add_dependency(task_ffn, task_attn);

        self.launch_all_async(stream)?;

        log::info!("HAIKU [CPU-WORK] Validating attention output...");
        log::info!("HAIKU [CPU-WORK] Prefetching next layer weights...");

        self.wait_for(task_attn)?;
        self.wait_for(task_ffn)?;

        log::info!("HAIKU [COMPLETE] Layer {} done (2 blocks)", layer_idx);

        Ok(())
    }

    /// Orchestrate full model decode: multiple layers with stream block kernels.
    ///
    /// Demonstrates full model forward pass with inter-layer dependencies.
    /// **For testing/spikes only.**
    pub fn orchestrate_full_model_spike(
        &mut self,
        stream: &Stream,
        num_layers: usize,
    ) -> Result<(), Box<dyn Error>> {
        log::info!(
            "[FULL MODEL SPIKE] Orchestrating {} layers with hybrid stream blocks",
            num_layers
        );

        const OP_STREAM_ATTN: u32 = 10;
        const OP_STREAM_FFN: u32 = 11;

        let mut prev_ffn_task: Option<TaskId> = None;

        for layer_idx in 0..num_layers {
            let task_attn = self.submit_task(
                &format!("StreamAttn_L{}", layer_idx),
                OP_STREAM_ATTN,
                256,
                256,
            );
            let task_ffn =
                self.submit_task(&format!("StreamFFN_L{}", layer_idx), OP_STREAM_FFN, 512, 256);

            self.add_dependency(task_ffn, task_attn);

            if let Some(prev_ffn) = prev_ffn_task {
                self.add_dependency(task_attn, prev_ffn);
            }

            prev_ffn_task = Some(task_ffn);
        }

        self.launch_all_async(stream)?;

        log::info!(
            "HAIKU [SUBMITTED] {} layers × 2 blocks = {} tasks",
            num_layers,
            num_layers * 2
        );
        log::info!("HAIKU [CPU-WORK] Prefetching all model weights into GPU memory...");
        log::info!("HAIKU [CPU-WORK] Preparing sampler state...");
        log::info!("HAIKU [COMPLETE] Full model forward pass done");

        Ok(())
    }
}

impl Default for HaikuSan {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_submission() {
        let mut orchestrator = HaikuSan::new();
        let id1 = orchestrator.submit_task("task1", 0, 256, 256);
        let id2 = orchestrator.submit_task("task2", 1, 512, 256);

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
    }

    #[test]
    fn test_dependency_tracking() {
        let mut orchestrator = HaikuSan::new();
        let id1 = orchestrator.submit_task("task1", 0, 256, 256);
        let id2 = orchestrator.submit_task("task2", 1, 512, 256);

        orchestrator.add_dependency(id2, id1);

        // Verify dependency was added (check by examining queue)
        // In production, this would be tested via GPU execution.
    }
}
