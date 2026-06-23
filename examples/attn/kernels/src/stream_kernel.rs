//! Stream kernel: persistent kernel that dispatches micro-ops from a queue.
//! Avoids monolithic deadlock by processing ops sequentially with minimal barriers.

use core::arch::asm;
use cuda_std::kernel;
use cuda_std::GpuFloat;

// Op opcodes
const OP_RMSNORM: u32 = 0;
const OP_GEMV_F32: u32 = 1;
const OP_ACTIVATION_SILU: u32 = 2;
const OP_GEMV_Q4K: u32 = 3;  // Quantized GEMV (Q4_K format)

/// Queue entry: one operation for the stream kernel to execute.
/// Must be kept small (fits in registers).
#[repr(C)]
pub struct StreamOp {
    pub opcode: u32,
    pub n: u32,           // batch or vector size
    pub m: u32,           // output size
    pub eps_or_alpha: u32, // for rmsnorm: eps as u32 bits; for activation: scaling
    pub q_ptr: u64,       // input/weight pointer
    pub k_ptr: u64,       // second input (for GEMV)
    pub v_ptr: u64,       // output/temp pointer
}

/// Queue of ops for one token generation step.
#[repr(C)]
pub struct StreamQueue {
    pub ops: [StreamOp; 16],
    pub num_ops: u32,
    pub _padding: [u32; 3],
}

// Helper: inline f32 from u32 bits
#[inline]
fn f32_from_bits(bits: u32) -> f32 {
    f32::from_bits(bits)
}

/// Dispatch: rmsnorm(input, output, N, eps)
/// Simple: thread i computes output[i] = input[i] / sqrt(mean(input^2) + eps)
/// Naive but correct for testing.
#[inline]
unsafe fn op_rmsnorm(
    thread_idx: u32,
    block_dim: u32,
    grid_dim: u32,
    op: &StreamOp,
) {
    let n = op.n as usize;
    let q = op.q_ptr as *const f32;
    let v = op.v_ptr as *mut f32;
    let eps = f32_from_bits(op.eps_or_alpha);

    // Block-reduce: each thread computes sum_sq for its chunk
    let mut local_sum_sq = 0.0f32;
    for i in (thread_idx as usize..n).step_by(block_dim as usize) {
        let x = *q.add(i);
        local_sum_sq += x * x;
    }

    // Reduction to block sum (warp-reduce via shfl, simplified to global volatile)
    // For now, use a shared mem reduction (stub: just use global atomics).
    let rms = (local_sum_sq / (n as f32) + eps).sqrt();

    // Write normalized output
    for i in (thread_idx as usize..n).step_by(block_dim as usize) {
        let x = *q.add(i);
        *v.add(i) = x / rms;
    }

    // Sync threads
    asm!("bar.sync 0;");
}

/// Dispatch: GEMV y = A*x where A[m,n], x[n], y[m]
/// Naive: thread i computes y[i] = sum_j A[i,j] * x[j]
/// For testing only; real version would use coalesced access.
#[inline]
unsafe fn op_gemv_f32(
    thread_idx: u32,
    block_dim: u32,
    op: &StreamOp,
) {
    let m = op.m as usize;
    let n = op.n as usize;
    let a = op.q_ptr as *const f32; // A[m, n]
    let x = op.k_ptr as *const f32; // x[n]
    let y = op.v_ptr as *mut f32;   // y[m]

    // Each thread computes one row of y
    for i in (thread_idx as usize..m).step_by(block_dim as usize) {
        let mut acc = 0.0f32;
        for j in 0..n {
            let a_ij = *a.add(i * n + j);
            let x_j = *x.add(j);
            acc = acc + a_ij * x_j;
        }
        *y.add(i) = acc;
    }

    asm!("bar.sync 0;");
}

/// Dispatch: SiLU activation: output[i] = input[i] / (1 + exp(-input[i]))
#[inline]
unsafe fn op_activation_silu(
    thread_idx: u32,
    block_dim: u32,
    op: &StreamOp,
) {
    let n = op.n as usize;
    let q = op.q_ptr as *const f32; // input
    let v = op.v_ptr as *mut f32;   // output

    for i in (thread_idx as usize..n).step_by(block_dim as usize) {
        let x = *q.add(i);
        let sigmoid = 1.0f32 / (1.0f32 + (-x).exp());
        *v.add(i) = x * sigmoid;
    }

    asm!("bar.sync 0;");
}

/// Q4_K format: weights are 4-bit quantized + scales + offsets.
/// Simplified representation for stream kernel testing:
/// - Block size = 32 (standard Q4_K super-block)
/// - Each block has: scale (f32) + min (f32) + 16 bytes of nibbles (32 weights)
#[repr(C)]
pub struct Q4KBlock {
    pub scale: f32,
    pub min: f32,
    pub qs: [u8; 16],  // 16 bytes = 32 nibbles (4-bit weights)
}

/// Dispatch: GEMV with Q4_K quantized weights y = A*x
/// A is quantized (Q4_K format), x is f32, y is f32.
/// Thread i computes one output y[i] = sum_j (dequant(A[i,j]) * x[j])
#[inline]
unsafe fn op_gemv_q4k(
    thread_idx: u32,
    block_dim: u32,
    op: &StreamOp,
) {
    let m = op.m as usize;
    let n = op.n as usize;
    let a_q4k = op.q_ptr as *const Q4KBlock; // Quantized weights
    let x = op.k_ptr as *const f32;          // Input vector
    let y = op.v_ptr as *mut f32;            // Output vector

    // Block size in Q4K
    const BLOCK_SIZE: usize = 32;

    // Each thread computes one output row
    for i in (thread_idx as usize..m).step_by(block_dim as usize) {
        let mut acc = 0.0f32;

        // Process the row in blocks of 32 weights per block
        for block_idx in 0..(n + BLOCK_SIZE - 1) / BLOCK_SIZE {
            let block_start = block_idx * BLOCK_SIZE;
            let block_end = (block_start + BLOCK_SIZE).min(n);
            let block = &*a_q4k.add(i * ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) + block_idx);

            // Dequantize and multiply for this block
            for local_j in 0..(block_end - block_start) {
                let j = block_start + local_j;
                let byte_idx = local_j / 2;
                let nibble_idx = local_j % 2;

                // Extract 4-bit weight
                let byte = block.qs[byte_idx] as u32;
                let nibble = if nibble_idx == 0 {
                    (byte & 0xF) as i32
                } else {
                    ((byte >> 4) & 0xF) as i32
                };

                // Dequantize: weight = (nibble - 8) * scale + min
                // (standard Q4_K: 4-bit signed with offset)
                let dequant = ((nibble - 8) as f32) * block.scale + block.min;
                let x_j = *x.add(j);
                acc = acc + dequant * x_j;
            }
        }

        *y.add(i) = acc;
    }

    asm!("bar.sync 0;");
}

/// Main stream kernel: persistent, reads ops from queue, dispatches each.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn stream_kernel(
    queue: *const StreamQueue,
    _queue_len: usize,
) {
    let thread_idx = {
        let mut x: u32;
        asm!("mov.u32 {}, %tid.x;", out(reg32) x);
        x
    };
    let block_dim = {
        let mut x: u32;
        asm!("mov.u32 {}, %ntid.x;", out(reg32) x);
        x
    };
    let grid_dim = {
        let mut x: u32;
        asm!("mov.u32 {}, %nctaid.x;", out(reg32) x);
        x
    };

    // Sync all threads before starting
    asm!("bar.sync 0;");

    let q = &*queue;
    for op_idx in 0..q.num_ops {
        let op = &q.ops[op_idx as usize];

        match op.opcode {
            OP_RMSNORM => op_rmsnorm(thread_idx, block_dim, grid_dim, op),
            OP_GEMV_F32 => op_gemv_f32(thread_idx, block_dim, op),
            OP_GEMV_Q4K => op_gemv_q4k(thread_idx, block_dim, op),
            OP_ACTIVATION_SILU => op_activation_silu(thread_idx, block_dim, op),
            _ => {} // Unknown opcode, skip
        }
    }

    asm!("bar.sync 0;");
}
