//! Role-based kernel dispatch for phase-aware execution
//!
//! Defines kernel roles (decode-optimized, prefill-optimized, etc.) and
//! provides dispatch logic to map opcodes to kernel implementations.

use crate::HaikuSan;
use std::error::Error;

/// Kernel role enumeration: identifies which specialized kernel to run
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelRole {
    // Decode phase roles (single-token, cache-optimized)
    /// RMS normalization for single row
    RmsnormSingle = 30,
    /// Thin GEMV (matrix-vector product)
    GemvDecodeSingle = 31,
    /// Single-query attention with cached K/V
    FlashAttnSingle = 32,

    // Prefill phase roles (batch-optimized)
    /// Vectorized RMS norm over batch
    RmsnormBatch = 20,
    /// GEMV with fused gate+up projection
    GemvBatchPrefill = 21,
    /// Multi-query attention over batch
    FlashAttnBatch = 22,

    // Sampler role
    /// On-device token sampling
    SamplerSingle = 33,

    // Advanced roles (future)
    /// Speculative decode attention (multi-hypothesis)
    SpecDecodeAttn = 40,
}

impl KernelRole {
    /// Convert opcode (u32) to KernelRole
    pub fn from_opcode(opcode: u32) -> Option<Self> {
        match opcode {
            30 => Some(KernelRole::RmsnormSingle),
            31 => Some(KernelRole::GemvDecodeSingle),
            32 => Some(KernelRole::FlashAttnSingle),
            20 => Some(KernelRole::RmsnormBatch),
            21 => Some(KernelRole::GemvBatchPrefill),
            22 => Some(KernelRole::FlashAttnBatch),
            33 => Some(KernelRole::SamplerSingle),
            40 => Some(KernelRole::SpecDecodeAttn),
            _ => None,
        }
    }

    /// Get human-readable name
    pub fn name(&self) -> &'static str {
        match self {
            KernelRole::RmsnormSingle => "RmsnormSingle",
            KernelRole::GemvDecodeSingle => "GemvDecodeSingle",
            KernelRole::FlashAttnSingle => "FlashAttnSingle",
            KernelRole::RmsnormBatch => "RmsnormBatch",
            KernelRole::GemvBatchPrefill => "GemvBatchPrefill",
            KernelRole::FlashAttnBatch => "FlashAttnBatch",
            KernelRole::SamplerSingle => "SamplerSingle",
            KernelRole::SpecDecodeAttn => "SpecDecodeAttn",
        }
    }

    /// Categorize by phase (decode vs prefill)
    pub fn phase(&self) -> Phase {
        match self {
            KernelRole::RmsnormSingle
            | KernelRole::GemvDecodeSingle
            | KernelRole::FlashAttnSingle
            | KernelRole::SamplerSingle => Phase::Decode,

            KernelRole::RmsnormBatch
            | KernelRole::GemvBatchPrefill
            | KernelRole::FlashAttnBatch => Phase::Prefill,

            KernelRole::SpecDecodeAttn => Phase::Decode, // Advanced decode
        }
    }
}

/// Execution phase (decode vs prefill)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    /// Single-token generation (decode)
    Decode,
    /// Multi-token prefill (batch processing)
    Prefill,
}

/// Role dispatch configuration for HaikuSan
///
/// Maps kernel roles to their configuration parameters (grid size, block size, etc.)
pub struct RoleDispatcher {
    phase: Option<Phase>,
}

impl RoleDispatcher {
    /// Create a new role dispatcher
    pub fn new() -> Self {
        RoleDispatcher { phase: None }
    }

    /// Set active phase
    pub fn set_phase(&mut self, phase: Phase) {
        self.phase = Some(phase);
    }

    /// Get current phase
    pub fn current_phase(&self) -> Option<Phase> {
        self.phase
    }

    /// Validate task is appropriate for current phase
    pub fn validate_role(&self, role: KernelRole) -> Result<(), Box<dyn Error>> {
        if let Some(current_phase) = self.phase {
            if role.phase() != current_phase {
                return Err(format!(
                    "Role {:?} (phase: {:?}) incompatible with current phase: {:?}",
                    role,
                    role.phase(),
                    current_phase
                )
                .into());
            }
        }
        Ok(())
    }
}

impl Default for RoleDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

/// Extension trait for HaikuSan to support role-based tasks
pub trait HaikuSanRoleExt {
    /// Submit a task with a kernel role
    fn submit_role(
        &mut self,
        role: KernelRole,
        name: &str,
        m: u32,
        n: u32,
    ) -> Result<crate::TaskId, Box<dyn Error>>;

    /// Get the role dispatcher
    fn dispatcher(&self) -> &RoleDispatcher;

    /// Get mutable role dispatcher
    fn dispatcher_mut(&mut self) -> &mut RoleDispatcher;
}

impl HaikuSanRoleExt for HaikuSan {
    fn submit_role(
        &mut self,
        role: KernelRole,
        name: &str,
        m: u32,
        n: u32,
    ) -> Result<crate::TaskId, Box<dyn Error>> {
        // Validate role is compatible with current phase
        self.dispatcher.validate_role(role)?;

        // Submit as regular task with role opcode
        let task_id = self.submit_task(name, role as u32, m, n);
        Ok(task_id)
    }

    fn dispatcher(&self) -> &RoleDispatcher {
        &self.dispatcher
    }

    fn dispatcher_mut(&mut self) -> &mut RoleDispatcher {
        &mut self.dispatcher
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_from_opcode() {
        assert_eq!(KernelRole::from_opcode(30), Some(KernelRole::RmsnormSingle));
        assert_eq!(KernelRole::from_opcode(31), Some(KernelRole::GemvDecodeSingle));
        assert_eq!(KernelRole::from_opcode(32), Some(KernelRole::FlashAttnSingle));
        assert_eq!(KernelRole::from_opcode(999), None);
    }

    #[test]
    fn test_role_phase_classification() {
        assert_eq!(KernelRole::RmsnormSingle.phase(), Phase::Decode);
        assert_eq!(KernelRole::GemvDecodeSingle.phase(), Phase::Decode);
        assert_eq!(KernelRole::RmsnormBatch.phase(), Phase::Prefill);
        assert_eq!(KernelRole::GemvBatchPrefill.phase(), Phase::Prefill);
    }

    #[test]
    fn test_role_dispatcher() {
        let mut dispatcher = RoleDispatcher::new();

        // No phase set initially
        assert_eq!(dispatcher.current_phase(), None);

        // Set decode phase
        dispatcher.set_phase(Phase::Decode);
        assert_eq!(dispatcher.current_phase(), Some(Phase::Decode));

        // Decode roles should validate
        assert!(dispatcher.validate_role(KernelRole::RmsnormSingle).is_ok());
        assert!(dispatcher.validate_role(KernelRole::GemvDecodeSingle).is_ok());

        // Prefill roles should fail
        assert!(dispatcher.validate_role(KernelRole::RmsnormBatch).is_err());
    }
}
