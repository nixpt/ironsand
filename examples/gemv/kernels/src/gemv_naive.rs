use cuda_std::kernel;
use cuda_std::thread;

/// Naive GEMV: `y = alpha * A·x + beta * y`.
///
/// One thread per output row. Each thread streams a full row of `A` (k elements)
/// and the whole of `x`. Simple and correct, but uncoalesced: adjacent threads
/// read rows `k` apart, so successive lanes touch far-apart memory.
///
/// # Safety
/// CUDA kernel; the host must launch with at least `m` threads and buffers sized
/// `A=m*k`, `x=k`, `y=m`.
///
/// # Parameters
/// - `a`: weight matrix, `m x k`, row-major.
/// - `x`: input vector, length `k`.
/// - `y`: output vector, length `m`. Read iff `beta != 0`, then written.
/// - `m`, `k`: dimensions.
/// - `alpha`, `beta`: scalars.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_naive(
    a: &[f32],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let row = (thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x()) as usize;
    if row < m {
        let base = row * k;
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += a[base + i] * x[i];
        }
        let elem = unsafe { &mut *y.add(row) };
        *elem = alpha * sum + beta * *elem;
    }
}
