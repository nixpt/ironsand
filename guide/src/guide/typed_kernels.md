# Typed Kernels

The [`Kernel<'a, Args>`](../../kernel/struct.Kernel.html) type provides a **compile-time-typed** handle to a GPU kernel. It encodes the kernel's parameter signature in the `Args` tuple, so every launch is checked by the Rust compiler. This eliminates an entire class of runtime errors — wrong argument count, wrong argument order, or passing a pointer where a scalar is expected.

This chapter covers everything from basic loading and launching to advanced topics such as occupancy-driven launch configuration and reusing a loaded kernel across multiple CUDA streams.

```text
 Device code                              Host code
 ───────────                              ─────────

 #[kernel]                                kernel_descriptor! {
 pub unsafe fn saxpy(                        pub unsafe fn saxpy(
     x: *const f32,     ───PTX───▶            x: DevicePointer<f32>,
     y: *mut f32,                             y: DevicePointer<f32>,
     a: f32,                                  a: f32,
     n: u32,                                  n: usize,
 );                                        ); }
     │                                          │
     │    1. Compile to PTX with cuda_builder   │
     ▼                                          ▼
 ┌─────────┐                              ┌──────────────┐
 │ PTX blob│  ◄──── 2. Load at runtime ───│   Module     │
 └─────────┘                              └──────────────┘
                                               │
                                               │ 3. Descriptor::load
                                               ▼
                                          ┌──────────────┐
                                          │ Kernel<Args> │
                                          └──────────────┘
                                               │
                                               │ 4. kernel.launch
                                               ▼
                                          ┌──────────────┐
                                          │   GPU exec   │
                                          └──────────────┘
```

The four steps are:

1. **Write** the kernel in Rust with `#[kernel]` and compile it to PTX via `cuda_builder`.
2. **Load** the PTX blob into a `cust::module::Module` at runtime.
3. **Describe** the kernel's host-side ABI with `kernel_descriptor!` (or `#[derive(KernelDescriptor)]`) and load a `Kernel<'a, Args>` from the module.
4. **Launch** with `kernel.launch(grid, block, shared_mem, &stream, args)` — the Rust compiler checks that `args` matches the descriptor's `Args` tuple.

## Quick recap

If you have not read the [Getting Started](getting_started.html) chapter yet, the typed API can be introduced in three lines:

```rs
use cust::prelude::*;

// 1. Describe the kernel's host-side ABI.
kernel_descriptor! {
    pub unsafe fn saxpy(
        x: DevicePointer<f32>,
        y: DevicePointer<f32>,
        a: f32,
        n: usize,
    );
}

// 2. Load it once from the module.
let saxpy = saxpy::load(&module)?;

// 3. Launch it — the tuple must exactly match the descriptor.
unsafe {
    saxpy.launch(256, 128, 0, &stream, (x_ptr, y_ptr, 2.0f32, 1024usize))?;
}
```

The same works with the derive macro:

```rs
#[derive(KernelDescriptor)]
#[kernel_name = "saxpy"]
struct Saxpy(DevicePointer<f32>, DevicePointer<f32>, f32, usize);

let saxpy = Saxpy::load(&module)?;
```

## Migrating from launch! to Kernel::launch

If you have existing code that uses the raw `launch!` macro and `module.get_function`, the migration is mechanical and usually takes only a few minutes per kernel.

### Step-by-step

1. **Remove** `use cust::launch;` and `module.get_function("...")`.
2. **Add** `use cust::kernel::KernelDescriptor;` and `use cust::kernel_descriptor;`.
3. **Declare** the kernel's host-side ABI with `kernel_descriptor!` (or derive it).
4. **Load** the kernel once with `Descriptor::load(&module)?`.
5. **Replace** `launch!(func<<<grid, block, smem, stream>>>(...))` with `kernel.launch(grid, block, smem, &stream, (...))`.

### Simple kernel

**Before** — raw `launch!`:

```rs
use cust::launch;
use cust::function::{BlockSize, GridSize};

let func = module.get_function("increment")?;
let blocks = BlockSize::xy(512, 1);
let grids = GridSize::xy(1024, 1);

unsafe {
    launch!(func<<<grids, blocks, 0, stream>>>(
        device_a.as_device_ptr(),
        value,
    ))?;
}
```

**After** — typed `Kernel`:

```rs
use cust::kernel::KernelDescriptor;
use cust::kernel_descriptor;
use cust::function::{BlockSize, GridSize};

kernel_descriptor! {
    pub unsafe fn increment(g_data: DevicePointer<u32>, inc_value: u32);
}

let increment = increment::load(&module)?;
let blocks = BlockSize::xy(512, 1);
let grids = GridSize::xy(1024, 1);

unsafe {
    increment.launch(grids, blocks, 0, &stream, (
        device_a.as_device_ptr(),
        value,
    ))?;
}
```

### Kernel with slices

Device-side slices (`&[T]`) become a `(DevicePointer<T>, usize)` pair on the host.

**Before**:

```rs
let func = module.get_function("matrix_mul_cuda")?;
unsafe {
    launch!(func<<<grid, threads, 0, stream>>>(
        d_c.as_device_ptr(),
        d_a.as_device_ptr(), d_a.len(),
        d_b.as_device_ptr(), d_b.len(),
        dims_a.x,
        dims_b.x,
    ))?;
}
```

**After**:

```rs
kernel_descriptor! {
    pub unsafe fn matrix_mul_cuda(
        c: DevicePointer<f32>,
        a: DevicePointer<f32>, a_len: usize,
        b: DevicePointer<f32>, b_len: usize,
        wa: usize,
        wb: usize,
    );
}

let matrix_mul_cuda = matrix_mul_cuda::load(&module)?;
unsafe {
    matrix_mul_cuda.launch(grid, threads, 0, &stream, (
        d_c.as_device_ptr(),
        d_a.as_device_ptr(), d_a.len(),
        d_b.as_device_ptr(), d_b.len(),
        dims_a.x,
        dims_b.x,
    ))?;
}
```

### Common pitfalls

| Issue | Raw API | Typed API |
|---|---|---|
| **Stream reference** | `launch!(func<<<..., stream>>>(...))` — stream is passed by value in the macro | `kernel.launch(..., &stream, (...))` — `launch` takes `&Stream`, so pass a reference |
| **Argument tuple** | Arguments are expanded directly in the macro | Arguments are wrapped in a Rust tuple: `(a, b, c)` |
| **`DevicePointer` import** | Only needed if you use it explicitly | Needed in the `kernel_descriptor!` declaration; add `use cust::memory::DevicePointer;` or rely on `cust::prelude::*` |
| **Loading** | `module.get_function("name")?` each time you need it | `Descriptor::load(&module)?` once, then reuse the `Kernel` handle |

### Gradual adoption

You do not have to migrate every kernel at once. A `Kernel` can be converted back to a raw `Function` with `.into()` or `.as_function()`, so you can wrap one kernel in the typed API while leaving the rest unchanged:

```rs
let typed = saxpy::load(&module)?;
let raw: Function = typed.into(); // hand off to legacy code
```

## Manual loading without a descriptor

Sometimes you need the type safety of `Kernel` but do not want to declare a descriptor (e.g. for a one-off experiment or a dynamically-named kernel). Use `Module::get_kernel`:

```rs
use cust::kernel::Kernel;

let k: Kernel<(DevicePointer<f32>, usize)> =
    module.get_kernel("scale")?;
```

If the tuple does not match the kernel's actual PTX signature, the launch will still be ABI-correct (because the `KernelArgs` trait writes the tuple fields in order) but the kernel will receive garbage values. Choose the tuple type carefully, referring to the [Kernel ABI](kernel_abi.html) chapter if you are unsure how device-side types map to host-side tuples.

## Launch configuration

`Kernel::launch` accepts any type that implements `Into<GridSize>` and `Into<BlockSize>`. The simplest form is plain integers:

```rs
unsafe {
    saxpy.launch(256, 128, 0, &stream, args)?;
}
```

For 2-D or 3-D grids you can pass tuples:

```rs
unsafe {
    saxpy.launch((16, 16), (32, 32), 0, &stream, args)?;
}
```

Or use the explicit `GridSize` and `BlockSize` builders:

```rs
use cust::function::{GridSize, BlockSize};

let grid = GridSize::xyz(16, 16, 1);
let block = BlockSize::xyz(32, 32, 1);
unsafe {
    saxpy.launch(grid, block, 0, &stream, args)?;
}
```

### Dynamic shared memory

The third numeric argument to `launch` is the number of **bytes** of dynamic shared memory to allocate per block. Set it to `0` if the kernel does not need extra dynamic shared memory beyond any `__shared__` arrays whose size is fixed at compile time.

```rs
unsafe {
    saxpy.launch(grid, block, 4096, &stream, args)?;
}
```

## Occupancy queries

`Kernel` forwards the CUDA occupancy API to the underlying [`Function`](../../function/struct.Function.html), so you can query optimal launch configurations without dropping back to raw handles.

### Suggested launch configuration

`suggested_launch_configuration` asks the driver for a block size that maximizes occupancy given a fixed amount of dynamic shared memory:

```rs
let (grid_size, block_size) =
    saxpy.suggested_launch_configuration(0, BlockSize::x(1024))?;

unsafe {
    saxpy.launch(grid_size, block_size, 0, &stream, args)?;
}
```

The first argument is dynamic shared memory in bytes; the second is an upper bound on block size. The returned `grid_size` is a `u32` and `block_size` is a `u32` — cast them to `GridSize`/`BlockSize` or pass them directly to `launch` (they implement `Into<BlockSize>` and `Into<GridSize>`).

### Maximum active blocks per SM

If you already know your block size and want to know how many blocks can run concurrently on each streaming multiprocessor:

```rs
let max_blocks = saxpy.max_active_blocks_per_multiprocessor(
    256.into(), // block size
    0,          // dynamic shared memory
)?;
```

### Function attributes

You can also read raw PTX attributes such as `.maxntid` and `.reqntid`:

```rs
use cust::function::FunctionAttribute;

let max_threads = saxpy.get_attribute(FunctionAttribute::MaxThreadsPerBlock)?;
let shared_size = saxpy.get_attribute(FunctionAttribute::SharedSizeBytes)?;
```

See [`FunctionAttribute`](../../function/enum.FunctionAttribute.html) for the full list.

## Reusing a kernel across streams

A `Kernel` is `Clone`, `Copy`, and cheap to duplicate — it is just a borrowed handle to a driver function object. The canonical pattern is to **load once** and **launch many times** on different streams or in a loop:

```rs
let saxpy = saxpy::load(&module)?;

let stream_a = Stream::new(StreamFlags::NON_BLOCKING, None)?;
let stream_b = Stream::new(StreamFlags::NON_BLOCKING, None)?;

unsafe {
    saxpy.launch(256, 128, 0, &stream_a, (x_a, y_a, 2.0, n))?;
    saxpy.launch(256, 128, 0, &stream_b, (x_b, y_b, 3.0, n))?;
}
```

Both launches share the same underlying function handle but execute in independent streams. Remember the safety rules from the [Safety](safety.html) chapter: writing to the same memory location from kernels in different streams without explicit synchronization is undefined behavior.

### Example: pipelined inference decode

A common decode-phase pattern is to enqueue one token's kernels while the previous token is still in flight:

```rs
let rms = rms_norm::load(&module)?;
let gemv = gemv::load(&module)?;
let attn = flash_attn::load(&module)?;

for i in 0..token_streams.len() {
    let stream = &streams[i % streams.len()];
    // SAFETY: each stream writes to its own distinct output slice.
    let out_i = &out_buffers[i];

    unsafe {
        rms.launch(1, 1024, 0, stream, (hidden.as_device_ptr(), hidden.len()))?;
        gemv.launch(1, 1024, 0, stream, (wq.as_device_ptr(), xq.as_device_ptr(), out_i.as_device_ptr(), m, n))?;
        attn.launch(1, 64, 0, stream, (q.as_device_ptr(), k_cache.as_device_ptr(), v_cache.as_device_ptr(), out_i.as_device_ptr(), head_dim, seq_len))?;
    }
}
```

Each `Kernel` handle is reused across all iterations; only the `Stream` changes.

## Raw handle fallback

If you need to interface with code that expects a raw [`Function`](../../function/struct.Function.html) (for example, a third-party wrapper or a legacy `launch!` call site), convert the typed handle with `.into()` or access the inner handle with `.as_function()`:

```rs
// Move the typed handle into a raw Function.
let func: Function = saxpy.into();
```

Or borrow the inner handle without consuming the `Kernel`:

```rs
let func_ref: &Function = saxpy.as_function();
```

This is useful for gradual adoption: you can wrap existing kernels in `Kernel` handles without rewriting all surrounding code at once.

## Choosing between `typed_kernel!`, `kernel_descriptor!`, and `#[derive(KernelDescriptor)]`

| Approach | Best for |
|---|---|
| `typed_kernel!` | One-off loading where you want to avoid spelling the full `Kernel<(... )>` type. |
| `kernel_descriptor!` | Kernels you launch repeatedly; gives a named struct with `::load`. Mirrors a function signature, so it is easy to keep in sync with the device code. |
| `#[derive(KernelDescriptor)]` | When you want a strongly-named type (e.g. `Saxpy`) that you can pass around as a generic parameter or store in a struct field. |

All three compile down to the same `Kernel<'a, Args>` handle at runtime; the difference is purely ergonomic.
