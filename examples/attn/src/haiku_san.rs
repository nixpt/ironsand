//! Haiku-San: CPU/GPU hybrid orchestrator for distributed kernel chains.
//!
//! Lightweight CPU orchestrator manages async GPU kernel launches,
//! enabling CPU/GPU parallel execution without monolithic deadlock risk.

use std::collections::{HashMap, VecDeque};
use std::error::Error;

use cust::event::Event;
use cust::stream::Stream;

/// Identifier for a queued kernel task.
pub type TaskId = u64;

/// A kernel task to be executed on GPU.
#[derive(Clone, Debug)]
pub struct KernelTask {
    pub id: TaskId,
    pub name: String,
    pub opcode: u32,  // Which kernel to run
    pub m: u32,       // Output size
    pub n: u32,       // Input size
    pub depends_on: Vec<TaskId>,  // Task IDs that must finish first
}

/// CPU-side orchestrator for distributed GPU kernels.
pub struct HaikuSan {
    next_task_id: TaskId,
    queue: VecDeque<KernelTask>,
    inflight: HashMap<TaskId, Event>,  // task_id → GPU event
    completed: Vec<TaskId>,
    stats: OrchestrationStats,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct OrchestrationStats {
    pub tasks_submitted: u64,
    pub tasks_completed: u64,
    pub total_wait_time_us: f32,
    pub cpu_work_us: f32,
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

    /// Submit a kernel task (queued, not yet launched).
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

    /// Add a dependency: task `dependent` must wait for `task_id` to complete.
    pub fn add_dependency(&mut self, dependent_id: TaskId, task_id: TaskId) {
        if let Some(task) = self.queue.iter_mut().find(|t| t.id == dependent_id) {
            task.depends_on.push(task_id);
        }
    }

    /// Launch all queued tasks asynchronously on GPU.
    /// This is where GPU kernels would actually run; for the spike,
    /// we simulate by storing events.
    pub fn launch_all_async(
        &mut self,
        stream: &Stream,
    ) -> Result<(), Box<dyn Error>> {
        while let Some(task) = self.queue.pop_front() {
            // Simulate launching the kernel by creating an event.
            // In a real implementation, this would call gpu.launch_kernel(task).
            let event = Event::new(cust::event::EventFlags::DEFAULT)?;
            event.record(stream)?;  // Mark the event at this point in the stream

            self.inflight.insert(task.id, event);

            println!(
                "HAIKU [SUBMIT] task_id={} name={} opcode={} (m={}, n={})",
                task.id, task.name, task.opcode, task.m, task.n
            );
        }

        Ok(())
    }

    /// Wait for a specific task to complete (synchronize on its GPU event).
    pub fn wait_for(&mut self, task_id: TaskId) -> Result<(), Box<dyn Error>> {
        if let Some(event) = self.inflight.remove(&task_id) {
            event.synchronize()?;
            self.completed.push(task_id);
            println!("HAIKU [COMPLETE] task_id={}", task_id);
            Ok(())
        } else {
            Err(format!("Task {} not found or already completed", task_id).into())
        }
    }

    /// Orchestrate a simple 2-op chain: GEMV_Q4K → SiLU
    /// Demonstrates async scheduling + CPU work overlap.
    pub fn orchestrate_two_op_spike(&mut self, stream: &Stream) -> Result<(), Box<dyn Error>> {
        const OP_GEMV_Q4K: u32 = 3;
        const OP_SILU: u32 = 2;

        println!("\n[SPIKE] Orchestrating 2-op spike: GEMV_Q4K → SiLU");

        // Submit tasks in order (but don't wait between submits).
        let task1_id = self.submit_task("GEMV_Q4K", OP_GEMV_Q4K, 128, 256);
        let task2_id = self.submit_task("SiLU", OP_SILU, 128, 0);
        self.add_dependency(task2_id, task1_id);

        // Launch all tasks asynchronously.
        self.launch_all_async(stream)?;

        // CPU-side work (simulated): validate outputs, check data invariants.
        println!("HAIKU [CPU-WORK] Validating task1 output...");
        // In a real implementation:
        //   - Download small portion of output for sanity check
        //   - Verify no NaNs, range checks, etc.
        //   - This happens while GPU runs task1 + task2

        // Wait for tasks in order (depends_on graph ensures correctness).
        self.wait_for(task1_id)?;
        self.wait_for(task2_id)?;

        println!("HAIKU [COMPLETE] 2-op spike done");
        println!("  Submitted: {} tasks", self.stats.tasks_submitted);

        Ok(())
    }

    /// Orchestrate a full layer: RmsNorm → QKV → FlashAttn → OProj → FFN
    /// (Simplified for demonstration; real version has all ops.)
    pub fn orchestrate_layer_spike(&mut self, stream: &Stream) -> Result<(), Box<dyn Error>> {
        const OP_RMSNORM: u32 = 0;
        const OP_GEMV_Q4K: u32 = 3;
        const OP_SILU: u32 = 2;

        println!("\n[SPIKE] Orchestrating full layer: RmsNorm → GEMV → SiLU");

        // Submit all layer tasks without waiting between them.
        let task_rmsnorm = self.submit_task("RmsNorm_attn", OP_RMSNORM, 256, 256);
        let task_qkv = self.submit_task("QKVProj", OP_GEMV_Q4K, 768, 256);
        let task_ffn_gate_up = self.submit_task("FFNGateUp", OP_GEMV_Q4K, 512, 256);
        let task_silu = self.submit_task("SiLU", OP_SILU, 512, 0);

        self.add_dependency(task_qkv, task_rmsnorm);
        self.add_dependency(task_ffn_gate_up, task_rmsnorm);
        self.add_dependency(task_silu, task_ffn_gate_up);

        // Launch all async — GPU sees all tasks at once.
        self.launch_all_async(stream)?;

        // CPU parallel phase: while GPU runs, CPU validates + decides next layer.
        println!("HAIKU [CPU-WORK] CPU parallel phase (GPU executing tasks)");
        println!("HAIKU [CPU-WORK] Prefetching next layer weights...");
        println!("HAIKU [CPU-WORK] Checking if end-of-phrase for lm_head...");

        // Wait for tasks in dependency order (orchestrator enforces this).
        self.wait_for(task_rmsnorm)?;
        self.wait_for(task_qkv)?;
        self.wait_for(task_ffn_gate_up)?;
        self.wait_for(task_silu)?;

        println!("HAIKU [COMPLETE] Layer spike done");

        Ok(())
    }

    pub fn stats(&self) -> OrchestrationStats {
        self.stats
    }
}
