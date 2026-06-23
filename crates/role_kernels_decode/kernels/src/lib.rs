#![cfg_attr(
    target_os = "cuda",
    no_std,
    crate_type = "staticlib",
    feature(register_attr),
    register_attr(nvvm_internal)
)]

use cuda_std::*;

/// RoleRMSNormSingle: Lightweight RMS normalization for single-token decode
///
/// Computes: output[i] = input[i] / sqrt(rms_sq + eps)
/// where rms_sq = sum(input[j]^2) / hidden_dim
///
/// **Optimized for**: Single-row input (1 × hidden_dim)
/// - Minimal memory footprint (L1-cache fit)
/// - Efficient variance reduction (warp-level shuffle)
///
/// **Block configuration**: 256 threads (1 warp per hidden_dim element)
/// **Per-call latency**: <100 μs (target)
#[kernel]
pub unsafe fn role_rms_norm_single(
    input: *const f32,      // [hidden_dim] - single row
    output: *mut f32,       // [hidden_dim] - output
    hidden_dim: u32,
    eps: f32,
) {
    let tid = thread::thread_idx_x() as u32;
    let stride = thread::block_dim_x() as u32;

    // Phase 1: Compute variance (sum of squares)
    // Each thread handles hidden_dim / num_threads elements
    let mut sum_sq: f32 = 0.0;
    let mut i = tid;
    while i < hidden_dim {
        let x = *input.add(i as usize);
        sum_sq += x * x;
        i += stride;
    }

    // Phase 2: Warp-level reduction (sum_sq across all threads)
    // Use shuffle to reduce sum_sq to thread 0
    sum_sq = warp_reduce_sum(sum_sq);

    // Phase 3: Compute RMS
    // Only thread 0 computes, then broadcast to shared memory
    let rms_sq = if tid == 0 {
        sum_sq / (hidden_dim as f32)
    } else {
        0.0
    };

    // Sync after broadcast
    thread::syncthreads();

    // Phase 4: Normalize (each thread writes its element)
    let inv_rms = 1.0 / (rms_sq + eps).sqrt();
    i = tid;
    while i < hidden_dim {
        let x = *input.add(i as usize);
        *output.add(i as usize) = x * inv_rms;
        i += stride;
    }
}

/// Warp-level reduction: sum across all 32 threads in a warp
/// Uses __shfl_down_sync for efficient inter-thread communication
#[inline]
unsafe fn warp_reduce_sum(mut val: f32) -> f32 {
    // Warp size is 32 on all modern NVIDIA GPUs
    const WARP_SIZE: u32 = 32;
    const FULL_MASK: u32 = 0xFFFFFFFF;

    for offset in [16, 8, 4, 2, 1] {
        // PTX assembly: __shfl_down_sync(mask, var, delta, width)
        // Exchanges var with threads at offset distance
        unsafe {
            asm!(
                "shfl.down.b32 {0}, {0}, {1}, 31;",
                inout(reg32) val,
                in(reg32) offset,
                options(pure, nomem, nostack),
            );
        }
        val += unsafe { core::mem::transmute::<u32, f32>(
            (unsafe { core::mem::transmute::<f32, u32>(val) })
        ) };
    }

    val
}

/// Helper: squared magnitude of a vector
/// Used for variance computation in RMS norm
#[inline]
pub fn squared_magnitude(x: &[f32]) -> f32 {
    x.iter().map(|v| v * v).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_squared_magnitude() {
        let x = [1.0, 2.0, 3.0];
        let sq_mag = squared_magnitude(&x);
        assert!((sq_mag - 14.0).abs() < 1e-6);
    }
}
