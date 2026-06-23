//! Integration tests for role-based Haiku-San orchestration

use haiku_san::{HaikuSan, HaikuSanRoleExt, KernelRole, Phase};

#[test]
fn test_role_based_decode_phase() {
    let mut orchestrator = HaikuSan::new();

    // Set decode phase
    orchestrator.dispatcher_mut().set_phase(Phase::Decode);
    assert_eq!(orchestrator.dispatcher().current_phase(), Some(Phase::Decode));

    // Submit decode roles
    let task1 = orchestrator
        .submit_role(KernelRole::RmsnormSingle, "RmsNorm_L0", 1, 8192)
        .expect("Should submit RmsnormSingle");

    let task2 = orchestrator
        .submit_role(
            KernelRole::GemvDecodeSingle,
            "GEMV_L0",
            8192,
            8192,
        )
        .expect("Should submit GemvDecodeSingle");

    let task3 = orchestrator
        .submit_role(
            KernelRole::FlashAttnSingle,
            "Attn_L0",
            1,
            1024,
        )
        .expect("Should submit FlashAttnSingle");

    // Add dependencies: task2 waits for task1, task3 waits for task2
    orchestrator.add_dependency(task2, task1);
    orchestrator.add_dependency(task3, task2);

    // Verify task IDs are sequential
    assert_eq!(task1, 0);
    assert_eq!(task2, 1);
    assert_eq!(task3, 2);
}

#[test]
fn test_role_phase_validation() {
    let mut orchestrator = HaikuSan::new();

    // Set decode phase
    orchestrator.dispatcher_mut().set_phase(Phase::Decode);

    // Decode roles should work
    assert!(
        orchestrator
            .submit_role(KernelRole::RmsnormSingle, "RmsNorm", 1, 8192)
            .is_ok()
    );

    // Prefill roles should fail in decode phase
    assert!(
        orchestrator
            .submit_role(KernelRole::RmsnormBatch, "RmsnormBatch", 32, 8192)
            .is_err()
    );
}

#[test]
fn test_role_based_prefill_phase() {
    let mut orchestrator = HaikuSan::new();

    // Set prefill phase
    orchestrator.dispatcher_mut().set_phase(Phase::Prefill);

    // Prefill roles should work
    assert!(
        orchestrator
            .submit_role(KernelRole::RmsnormBatch, "RmsnormBatch_L0", 32, 8192)
            .is_ok()
    );

    // Decode roles should fail in prefill phase
    assert!(
        orchestrator
            .submit_role(KernelRole::RmsnormSingle, "RmsnormSingle", 1, 8192)
            .is_err()
    );
}

#[test]
fn test_multi_layer_decode_orchestration() {
    let mut orchestrator = HaikuSan::new();
    orchestrator.dispatcher_mut().set_phase(Phase::Decode);

    // Orchestrate 2 layers with dependencies
    for layer in 0..2 {
        let prefix = format!("L{}", layer);

        let rms = orchestrator
            .submit_role(
                KernelRole::RmsnormSingle,
                &format!("{}_RmsNorm", prefix),
                1,
                8192,
            )
            .expect("submit rms");

        let gemv = orchestrator
            .submit_role(
                KernelRole::GemvDecodeSingle,
                &format!("{}_GEMV", prefix),
                8192,
                8192,
            )
            .expect("submit gemv");

        let attn = orchestrator
            .submit_role(
                KernelRole::FlashAttnSingle,
                &format!("{}_Attn", prefix),
                1,
                1024,
            )
            .expect("submit attn");

        // Intra-layer dependencies
        orchestrator.add_dependency(gemv, rms);
        orchestrator.add_dependency(attn, gemv);

        // Inter-layer: Layer N attention depends on Layer N-1 attention
        if layer > 0 {
            let prev_layer_task_id = (layer - 1) * 3 + 2; // Previous layer's attention task
            orchestrator.add_dependency(rms, prev_layer_task_id);
        }
    }

    // Verify stats
    let stats = orchestrator.stats();
    assert_eq!(stats.tasks_submitted, 0); // No launch yet
}

#[test]
fn test_kernel_role_names() {
    assert_eq!(KernelRole::RmsnormSingle.name(), "RmsnormSingle");
    assert_eq!(KernelRole::GemvDecodeSingle.name(), "GemvDecodeSingle");
    assert_eq!(KernelRole::FlashAttnSingle.name(), "FlashAttnSingle");
    assert_eq!(KernelRole::RmsnormBatch.name(), "RmsnormBatch");
}

#[test]
fn test_opcode_to_role_mapping() {
    use haiku_san::roles::KernelRole as Role;

    // Decode roles
    assert_eq!(Role::from_opcode(30), Some(KernelRole::RmsnormSingle));
    assert_eq!(Role::from_opcode(31), Some(KernelRole::GemvDecodeSingle));
    assert_eq!(Role::from_opcode(32), Some(KernelRole::FlashAttnSingle));

    // Prefill roles
    assert_eq!(Role::from_opcode(20), Some(KernelRole::RmsnormBatch));
    assert_eq!(Role::from_opcode(21), Some(KernelRole::GemvBatchPrefill));
    assert_eq!(Role::from_opcode(22), Some(KernelRole::FlashAttnBatch));

    // Invalid
    assert_eq!(Role::from_opcode(999), None);
}
