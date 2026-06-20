use cuda_std::kernel;
use cuda_std::thread;
use cuda_std::warp;

/// Warp size on all current NVIDIA hardware.
const WARP: u32 = 32;

/// Warp-per-row GEMV: `y = alpha * A·x + beta * y`.
///
/// Each warp owns one output row. The 32 lanes stride across `k` (lane `l`
/// reads columns `l, l+32, l+64, …`), so consecutive lanes touch consecutive
/// `A` elements — coalesced, unlike [`super::gemv_naive`]. Lane partials are
/// combined with a butterfly warp reduction, then lane 0 writes `y[row]`.
///
/// The host must launch with `block_dim_x` a multiple of 32 so hardware warps
/// align to `tid / 32` (and thus to a single output row).
///
/// # Safety
/// CUDA kernel; buffers sized `A=m*k`, `x=k`, `y=m`, and `>= m` warps launched.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_warp(
    a: &[f32],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    let tid = thread::block_dim_x() * thread::block_idx_x() + thread::thread_idx_x();
    let row = (tid / WARP) as usize; // global warp id == output row
    let lane = tid % WARP;
    if row >= m {
        // Whole warp exits together (all lanes share `row`), so the active warps
        // below always have a full 32-lane mask.
        return;
    }

    let base = row * k;
    let mut partial = 0.0f32;
    let mut i = lane as usize;
    while i < k {
        partial += a[base + i] * x[i];
        i += WARP as usize;
    }

    // Butterfly all-reduce over f32 *bits* — the shuffle intrinsics are integer
    // only (and cast lossily), so we move the bit pattern, not the value.
    let mask = u32::MAX;
    let mut offset = WARP / 2;
    while offset >= 1 {
        let (bits, _) = unsafe { warp::warp_shuffle_xor(mask, partial.to_bits(), offset, WARP) };
        partial += f32::from_bits(bits);
        offset >>= 1;
    }

    if lane == 0 {
        let elem = unsafe { &mut *y.add(row) };
        *elem = alpha * partial + beta * *elem;
    }
}
