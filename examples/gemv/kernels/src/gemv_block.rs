use core::mem::MaybeUninit;
use cuda_std::address_space;
use cuda_std::kernel;
use cuda_std::thread;

/// Threads per block (one block per output row).
pub const BLOCK: usize = 256;

/// Block-per-row GEMV: `y = alpha * A·x + beta * y`.
///
/// Each block owns one output row; its `BLOCK` threads stride across `k`
/// (thread `t` reads columns `t, t+BLOCK, …`), so the warps within the block
/// read `A` coalesced. Partial sums are combined with a shared-memory tree
/// reduction, then thread 0 writes `y[row]`.
///
/// This is the warp-shuffle reduction's portable cousin: the shuffle
/// intrinsics currently crash libnvvm during PTX generation (see the
/// `warp_shuffle` trap), so we reduce through shared memory — the same idiom
/// `gemm_tiled` uses — which lowers cleanly.
///
/// # Safety
/// CUDA kernel; launch with `grid = m`, `block = BLOCK`, buffers `A=m*k`,
/// `x=k`, `y=m`.
#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_block(
    a: &[f32],
    x: &[f32],
    y: *mut f32,
    m: usize,
    k: usize,
    alpha: f32,
    beta: f32,
) {
    #[address_space(shared)]
    static mut SCRATCH: [MaybeUninit<f32>; BLOCK] = [MaybeUninit::uninit(); BLOCK];

    // All threads in the block share one row, so the early-exit and every
    // `sync_threads` below are uniform across the block.
    let row = thread::block_idx_x() as usize;
    if row >= m {
        return;
    }
    let tid = thread::thread_idx_x() as usize;
    let base = row * k;

    let mut partial = 0.0f32;
    let mut i = tid;
    while i < k {
        partial += a[base + i] * x[i];
        i += BLOCK;
    }
    unsafe {
        SCRATCH[tid].write(partial);
    }
    thread::sync_threads();

    let mut stride = BLOCK / 2;
    while stride >= 1 {
        if tid < stride {
            let v = unsafe { SCRATCH[tid].assume_init() + SCRATCH[tid + stride].assume_init() };
            unsafe {
                SCRATCH[tid].write(v);
            }
        }
        thread::sync_threads();
        stride >>= 1;
    }

    if tid == 0 {
        let sum = unsafe { SCRATCH[0].assume_init() };
        let elem = unsafe { &mut *y.add(row) };
        *elem = alpha * sum + beta * *elem;
    }
}
