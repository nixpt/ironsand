<div align="center">
  <h1>ironsand</h1>

  <p>
    <strong>An ecosystem of libraries and tools for writing and executing extremely fast GPU code
    fully in <a href="https://www.rust-lang.org/">Rust</a>.</strong>
  </p>
</div>

## Purpose

`ironsand` is an experimentation vehicle for the **`zorro` inference engine**. The goal is to
explore authoring GPU kernels for LLM inference — GEMM, attention, sampling — in **Rust**, compiled
to PTX via the `rustc_codegen_nvvm` backend, as an alternative to hand-written CUDA C++.

This is a deliberately narrow scope. It is **not** a general-purpose GPU ecosystem (that is what its
upstream is); see below.

## Lineage & attribution

`ironsand` is a **hard fork** of [The Rust CUDA Project](https://github.com/Rust-GPU/rust-cuda)
(`Rust-GPU/rust-cuda`). The bulk of the code originates there and remains under its authors'
copyright (MIT OR Apache-2.0). Upstream targets general-purpose GPU work; this fork is slimmed to
the codegen + host stack needed for inference experiments, dropping OptiX, cuDNN, and non-inference
examples/infra. Full details and credits in [ATTRIBUTION.md](ATTRIBUTION.md).

This is an independent fork with its own history. It is **early and experimental**: expect bugs,
safety issues, and rough edges.

## Status

- Verified end-to-end (Rust kernel → PTX → executed on an sm_120 GPU) against **LLVM 19** + CUDA 13.3.
- The codegen backend's C++ shim is pinned to the LLVM 7 / 19 API (legacy Pass Manager). Newer LLVM
  (21/22) requires a real port, not a flag.

## Building the codegen backend

The `rustc_codegen_nvvm` backend needs a matching LLVM toolchain — major version **19** (with the
`llvm19` cargo feature) or 7 (default). Point it at an LLVM 19 install and build:

```sh
export CUDA_PATH=/path/to/cuda CUDA_ROOT=$CUDA_PATH CUDA_HOME=$CUDA_PATH
export LLVM_CONFIG_19=/path/to/llvm-19/bin/llvm-config
cargo build -p rustc_codegen_nvvm --features llvm19
```

To compile a kernel crate, enable the backend features on its `cuda_builder` build-dependency:

```toml
[build-dependencies]
cuda_builder = { workspace = true, default-features = false, features = ["rustc_codegen_nvvm", "llvm19"] }
```

## Typed Kernel API (recommended)

`cust` provides a compile-time-typed kernel handle, `Kernel<'a, Args>`, that encodes a kernel's
parameter signature in the Rust type system. This catches wrong argument counts, wrong types, and
wrong order at compile time rather than at runtime.

**Before** — raw `launch!` macro (unchecked at compile time):

```rust
let func = module.get_function("vecadd")?;
unsafe {
    launch!(func<<<256, 128, 0, stream>>>(a, a_len, b, b_len, c))?;
}
```

**After** — typed `Kernel` with a descriptor:

```rust
kernel_descriptor! {
    pub unsafe fn vecadd(
        a: DevicePointer<f32>, a_len: usize,
        b: DevicePointer<f32>, b_len: usize,
        c: DevicePointer<f32>
    );
}

let vecadd = vecadd::load(&module)?;
unsafe {
    vecadd.launch(256, 128, 0, &stream, (a, a_len, b, b_len, c))?;
}
```

Benefits:

- **Compile-time safety**: the tuple passed to `launch` must exactly match the descriptor's `Args` type.
- **Self-documenting**: the host-side declaration mirrors the device-side signature, making it obvious how to call the kernel.
- **Occupancy queries**: `Kernel` forwards `suggested_launch_configuration`, `max_active_blocks_per_multiprocessor`, and `get_attribute` directly.
- **Zero-cost**: a `Kernel` is just a phantom-type wrapper around `Function`; it compiles away.

See the [Typed Kernels guide chapter](guide/src/guide/typed_kernels.md) for full details on
`kernel_descriptor!`, `#[derive(KernelDescriptor)]`, multi-stream reuse, and raw-handle fallback.

### Migrating from `launch!`

If you have existing code using the raw `launch!` macro, the migration is mechanical:

1. **Replace** `module.get_function("name")` with a `kernel_descriptor!` declaration and `Descriptor::load(&module)`.
2. **Replace** `launch!(func<<<grid, block, 0, stream>>>(a, b, c))` with `kernel.launch(grid, block, 0, &stream, (a, b, c))`.
3. **Remove** `use cust::launch;`.

**Before**:
```rust
let func = module.get_function("increment")?;
unsafe {
    launch!(func<<<grids, blocks, 0, stream>>>(ptr, value))?;
}
```

**After**:
```rust
kernel_descriptor! {
    pub unsafe fn increment(ptr: DevicePointer<u32>, value: u32);
}

let increment = increment::load(&module)?;
unsafe {
    increment.launch(grids, blocks, 0, &stream, (ptr, value))?;
}
```

Device-side slices (`&[T]`) become `(DevicePointer<T>, usize)` pairs on the host. See the
[Migration guide](guide/src/guide/typed_kernels.md#migrating-from-launch-to-kernellaunch) in the
Typed Kernels chapter for slice examples, common pitfalls, and gradual-adoption tips.

## Documentation

The original [Rust CUDA Guide](https://rust-gpu.github.io/rust-cuda/) remains the best reference for
the shared architecture while `ironsand`'s own docs are written. See `guide/` in this repo.

## License

Inherited from the upstream project and unchanged. Licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
  http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your discretion.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in
the work by you, as defined in the Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
