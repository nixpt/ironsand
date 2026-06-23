//! Example: Orchestrating a decode phase with role-based kernels
//!
//! Demonstrates how to use Haiku-San to orchestrate a single token generation
//! across multiple layers using role-based decode kernels.

use haiku_san::{HaikuSan, HaikuSanRoleExt, KernelRole, Phase};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Haiku-San: Role-Based Decode Phase Orchestration ===\n");

    // Model parameters
    const NUM_LAYERS: usize = 32;
    const HIDDEN_DIM: u32 = 8192;
    const CACHE_LEN: u32 = 1024;

    // Create orchestrator
    let mut orchestrator = HaikuSan::new();

    // Set decode phase (single-token generation)
    orchestrator.dispatcher_mut().set_phase(Phase::Decode);
    println!("✓ Phase set to: Decode (single-token generation)\n");

    println!("Orchestrating {} layers with decode roles...", NUM_LAYERS);

    // Track task IDs for inter-layer dependencies
    let mut prev_layer_attn_task = None;

    // Orchestrate each layer
    for layer in 0..NUM_LAYERS {
        let layer_name = format!("L{}", layer);

        // Layer 1: RMS Norm
        let rms_task = orchestrator.submit_role(
            KernelRole::RmsnormSingle,
            &format!("{}_RmsNorm", layer_name),
            1,           // 1 token
            HIDDEN_DIM,  // hidden dimension
        )?;

        // Layer 2: QKV Projection (GEMV)
        let qkv_task = orchestrator.submit_role(
            KernelRole::GemvDecodeSingle,
            &format!("{}_QKV", layer_name),
            HIDDEN_DIM,  // output: qkv
            HIDDEN_DIM,  // input: hidden
        )?;
        orchestrator.add_dependency(qkv_task, rms_task); // qkv after rms_norm

        // Layer 3: Attention (uses cached K/V)
        let attn_task = orchestrator.submit_role(
            KernelRole::FlashAttnSingle,
            &format!("{}_Attn", layer_name),
            1,            // 1 query
            CACHE_LEN,    // seq_len of cache
        )?;
        orchestrator.add_dependency(attn_task, qkv_task); // attn after qkv

        // Layer 4: Output Projection (GEMV)
        let proj_task = orchestrator.submit_role(
            KernelRole::GemvDecodeSingle,
            &format!("{}_Proj", layer_name),
            HIDDEN_DIM,  // output: hidden
            HIDDEN_DIM,  // input: attn
        )?;
        orchestrator.add_dependency(proj_task, attn_task); // proj after attn

        // Inter-layer dependency: previous layer's projection → this layer's RMS norm
        if let Some(prev_attn) = prev_layer_attn_task {
            orchestrator.add_dependency(rms_task, prev_attn);
        }

        prev_layer_attn_task = Some(attn_task);

        if (layer + 1) % 8 == 0 {
            println!("  Layer {}: {} tasks queued", layer + 1, (layer + 1) * 4);
        }
    }

    println!("✓ All {} layers orchestrated", NUM_LAYERS);
    println!();

    // Show orchestration plan
    println!("=== Orchestration Plan ===");
    println!("Decode phase with {} layers", NUM_LAYERS);
    println!("Kernels per layer: 4 (RmsNorm + QKV + Attn + Proj)");
    println!("Total tasks: {} (no launch yet)", NUM_LAYERS * 4);
    println!("Hidden dimension: {}", HIDDEN_DIM);
    println!("Cache sequence length: {}", CACHE_LEN);
    println!();

    // Explain dependencies
    println!("=== Task Dependencies ===");
    println!("Intra-layer:");
    println!("  1. RmsNorm → QKV (normalize before projection)");
    println!("  2. QKV → Attention (Q from QKV projection)");
    println!("  3. Attention → Proj (attention output to final projection)");
    println!();
    println!("Inter-layer:");
    println!("  Layer N's RmsNorm → Layer N-1's Attention");
    println!("  (Previous layer's output feeds into next layer's input)");
    println!();

    // Expected characteristics
    println!("=== Expected Characteristics ===");
    println!("Total kernels: {} ({} per layer)", NUM_LAYERS * 4, 4);
    println!("CPU overhead: ~{} μs (64 kernels × 5 μs)", NUM_LAYERS * 4 * 5);
    println!("GPU execution: ~{} ms (realistic estimate)", NUM_LAYERS * 6 / 1000 + 1);
    println!("Speedup vs per-kernel: ~2-3× (reduced syncs + optimized kernels)");
    println!();

    // Statistics
    let stats = orchestrator.stats();
    println!("=== Orchestrator Stats ===");
    println!("Tasks submitted: {}", stats.tasks_submitted);
    println!("Tasks completed: {}", stats.tasks_completed);
    println!();

    println!("✓ Orchestration ready for GPU launch");
    println!("  Call: orchestrator.launch_all_async(&stream)?;");
    println!("  Then: orchestrator.wait_for(final_task_id)?;");

    Ok(())
}
