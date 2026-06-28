//! Role-Based Decode Kernels: Optimized GPU kernels for decode phase (single-token inference)
//!
//! This crate provides specialized GPU kernels for the decode phase of large language model
//! inference, where you're generating one token at a time. These kernels are optimized for
//! different compute patterns than prefill kernels.
//!
//! ## Decode Phase Characteristics
//!
//! - **Single token**: 1 output token at a time (not a batch)
//! - **KV cache hits**: Attention reuses cached key/value from prior tokens
//! - **Memory bandwidth bound**: Not compute bound
//! - **Lightweight operations**: RMS norm is L1-cache-fit, GEMV is thin
//!
//! ## Kernels
//!
//! - **RoleRMSNormSingle**: <100 μs, single-row normalization
//! - **RoleGEMVDecodeSingle**: <500 μs, thin matrix-vector product (8192→8192)
//! - **RoleFlashAttnSingle**: <2000 μs, single-query attention with cached K/V (1024-token cache)
//!
//! ## Integration with Haiku-San
//!
//! Each kernel is submitted to Haiku-San via an opcode:
//! ```ignore
//! const OP_RMSNORM_SINGLE: u32 = 30;
//! const OP_GEMV_DECODE_SINGLE: u32 = 31;
//! const OP_FLASH_ATTN_SINGLE: u32 = 32;
//!
//! let mut orchestrator = HaikuSan::new();
//! let task1 = orchestrator.submit_task("RmsNorm", OP_RMSNORM_SINGLE, 1, hidden_dim);
//! let task2 = orchestrator.submit_task("GEMV", OP_GEMV_DECODE_SINGLE, hidden_dim, hidden_dim);
//! orchestrator.add_dependency(task2, task1);
//! orchestrator.launch_all_async(&stream)?;
//! ```
//!
//! ## Design Files
//!
//! See `doc/ROLE_KERNELS_DESIGN.md` for:
//! - Phase-aware specialization rationale
//! - Register pressure analysis
//! - Memory coalescing patterns
//! - Expected performance gains

use cust::kernel::KernelDescriptor;
use cust::kernel_descriptor;
use cust::memory::{DeviceBuffer, DevicePointer};
use cust::module::Module;
use cust::stream::Stream;
use std::error::Error;

pub mod cpu_reference;

/// Opcode for RoleRMSNormSingle kernel (Haiku-San dispatcher)
pub const OP_RMSNORM_SINGLE: u32 = 30;
/// Opcode for RoleGEMVDecodeSingle kernel (Haiku-San dispatcher)
pub const OP_GEMV_DECODE_SINGLE: u32 = 31;
/// Opcode for RoleFlashAttnSingle kernel (Haiku-San dispatcher)
pub const OP_FLASH_ATTN_SINGLE: u32 = 32;

/// Launcher for RoleGEMVDecodeSingle kernel
///
/// # Arguments
/// * `module` - CUDA module with compiled kernels
/// * `stream` - CUDA stream for kernel execution
/// * `matrix` - Input matrix [m × n] (row-major)
/// * `vector` - Input vector [n]
/// * `output` - Output vector [m] (allocated by caller)
/// * `m` - Number of rows
/// * `n` - Number of columns
///
/// # Performance Target
/// - Latency: <500 μs (8192×8192)
/// - Memory: Coalesced access (row-major layout)
///
/// # Example
/// ```ignore
/// let m = 8192;
/// let n = 8192;
/// let mut matrix = vec![1.0; m * n];
/// let mut vector = vec![2.0; n];
/// let mut output = vec![0.0; m];
///
/// let dev_matrix = matrix.as_slice().as_device_boxed()?;
/// let dev_vector = vector.as_slice().as_device_boxed()?;
/// let mut dev_output = DeviceBuffer::zeroed(m)?;
///
/// launch_role_gemv_decode_single(
///     &module,
///     &stream,
///     &dev_matrix,
///     &dev_vector,
///     &mut dev_output,
///     m as u32,
///     n as u32,
/// )?;
/// ```
kernel_descriptor! {
    pub unsafe fn role_gemv_decode_single(
        matrix: DevicePointer<f32>,
        vector: DevicePointer<f32>,
        output: DevicePointer<f32>,
        m: u32,
        n: u32,
    );
}

pub fn launch_role_gemv_decode_single(
    module: &Module,
    stream: &Stream,
    matrix: &DeviceBuffer<f32>,
    vector: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    m: u32,
    n: u32,
) -> Result<(), Box<dyn Error>> {
    let kernel = role_gemv_decode_single::load(module)?;

    // Grid: m blocks (one block per output row)
    // Block: 256 threads (each computes partial dot product)
    const BLOCK_SIZE: u32 = 256;
    let grid_size = m;

    unsafe {
        kernel.launch(
            grid_size, BLOCK_SIZE, 0, stream,
            (
                matrix.as_device_ptr(),
                vector.as_device_ptr(),
                output.as_device_ptr(),
                m,
                n,
            ),
        )?;
    }

    Ok(())
}

/// Launcher for RoleRMSNormSingle kernel
///
/// # Arguments
/// * `module` - CUDA module with compiled kernels
/// * `stream` - CUDA stream for kernel execution
/// * `input` - Input vector [hidden_dim]
/// * `output` - Output vector [hidden_dim] (allocated by caller)
/// * `hidden_dim` - Vector dimension
/// * `eps` - Small epsilon for numerical stability (typically 1e-6)
///
/// # Performance Target
/// - Latency: <100 μs
/// - Memory: L1-cache fit (~32 KB for 8192 elements)
///
/// # Example
/// ```ignore
/// let hidden_dim = 4096;
/// let mut input = vec![1.0; hidden_dim];
/// let mut output = vec![0.0; hidden_dim];
///
/// let dev_input = input.as_slice().as_device_boxed()?;
/// let mut dev_output = DeviceBuffer::zeroed(hidden_dim)?;
///
/// launch_role_rms_norm_single(
///     &module,
///     &stream,
///     &dev_input,
///     &mut dev_output,
///     hidden_dim as u32,
///     1e-6,
/// )?;
/// ```
kernel_descriptor! {
    pub unsafe fn role_rms_norm_single(
        input: DevicePointer<f32>,
        output: DevicePointer<f32>,
        hidden_dim: u32,
        eps: f32,
    );
}

pub fn launch_role_rms_norm_single(
    module: &Module,
    stream: &Stream,
    input: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    hidden_dim: u32,
    eps: f32,
) -> Result<(), Box<dyn Error>> {
    let kernel = role_rms_norm_single::load(module)?;

    // Block size: 256 threads (one thread per ~(hidden_dim/256) elements)
    // This allows warp-level reductions to work efficiently
    const BLOCK_SIZE: u32 = 256;
    let grid_size = 1; // Single block per kernel call (fine for 1D RMS norm)

    unsafe {
        kernel.launch(
            grid_size, BLOCK_SIZE, 0, stream,
            (
                input.as_device_ptr(),
                output.as_device_ptr(),
                hidden_dim,
                eps,
            ),
        )?;
    }

    Ok(())
}

/// Launcher for RoleFlashAttnSingle kernel
///
/// # Arguments
/// * `module` - CUDA module with compiled kernels
/// * `stream` - CUDA stream for kernel execution
/// * `query` - Query vector [head_dim]
/// * `k_cache` - Cached keys [seq_len × head_dim]
/// * `v_cache` - Cached values [seq_len × head_dim]
/// * `output` - Output vector [head_dim] (allocated by caller)
/// * `head_dim` - Dimension of attention head
/// * `seq_len` - Length of cached sequence
///
/// # Performance Target
/// - Latency: <2000 μs (1024-token cache)
/// - Memory: All reads from cache (no K/V computation)
///
/// # Example
/// ```ignore
/// let head_dim = 128;
/// let seq_len = 1024;
/// let query = vec![1.0; head_dim];
/// let k_cache = vec![2.0; seq_len * head_dim];
/// let v_cache = vec![3.0; seq_len * head_dim];
/// let mut output = vec![0.0; head_dim];
///
/// let dev_query = query.as_slice().as_device_boxed()?;
/// let dev_k_cache = k_cache.as_slice().as_device_boxed()?;
/// let dev_v_cache = v_cache.as_slice().as_device_boxed()?;
/// let mut dev_output = DeviceBuffer::zeroed(head_dim)?;
///
/// launch_role_flash_attn_single(
///     &module,
///     &stream,
///     &dev_query,
///     &dev_k_cache,
///     &dev_v_cache,
///     &mut dev_output,
///     head_dim as u32,
///     seq_len as u32,
/// )?;
/// ```
kernel_descriptor! {
    pub unsafe fn role_flash_attn_single(
        query: DevicePointer<f32>,
        k_cache: DevicePointer<f32>,
        v_cache: DevicePointer<f32>,
        output: DevicePointer<f32>,
        head_dim: u32,
        seq_len: u32,
    );
}

pub fn launch_role_flash_attn_single(
    module: &Module,
    stream: &Stream,
    query: &DeviceBuffer<f32>,
    k_cache: &DeviceBuffer<f32>,
    v_cache: &DeviceBuffer<f32>,
    output: &mut DeviceBuffer<f32>,
    head_dim: u32,
    seq_len: u32,
) -> Result<(), Box<dyn Error>> {
    let kernel = role_flash_attn_single::load(module)?;

    // Block: 32 threads (warp, good for attention operations)
    // Grid: 1 (single query, single output)
    const BLOCK_SIZE: u32 = 32;
    let grid_size = 1;

    unsafe {
        kernel.launch(
            grid_size, BLOCK_SIZE, 0, stream,
            (
                query.as_device_ptr(),
                k_cache.as_device_ptr(),
                v_cache.as_device_ptr(),
                output.as_device_ptr(),
                head_dim,
                seq_len,
            ),
        )?;
    }

    Ok(())
}

/// Statistics about a kernel execution
#[derive(Clone, Copy, Debug)]
pub struct KernelStats {
    /// Kernel latency (milliseconds)
    pub latency_ms: f32,
    /// GPU occupancy (percentage)
    pub occupancy: u8,
    /// Memory bandwidth utilization (percentage)
    pub memory_bw_util: u8,
}

/// Kernel registry: Maps opcode to launcher function
pub struct KernelRegistry {
    module: Module,
}

impl KernelRegistry {
    /// Load all decode-role kernels from PTX
    pub fn load(device: &cust::device::Device) -> Result<Self, Box<dyn Error>> {
        let ptx = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
        let module = Module::from_ptx(ptx, &[])?;

        Ok(KernelRegistry { module })
    }

    /// Get the compiled module
    pub fn module(&self) -> &Module {
        &self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_rms_norm_reference() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = cpu_reference::rms_norm_single(&input, 1e-6);

        // Compute expected: rms_sq = (1 + 4 + 9 + 16) / 4 = 7.5
        // inv_rms = 1 / sqrt(7.5) ≈ 0.3651
        let expected = vec![0.3651, 0.7303, 1.0954, 1.4606];

        for (out, exp) in output.iter().zip(expected.iter()) {
            assert!(
                (out - exp).abs() < 1e-3,
                "Output mismatch: {} vs {}",
                out,
                exp
            );
        }
    }
}
