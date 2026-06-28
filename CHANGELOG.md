# Changelog

Notable changes to the ironsand project are documented in this file.

## Unreleased

### Typed Kernel API

A compile-time-typed kernel launch system has been added to `cust`, making kernel loading and launching safer and more ergonomic.

#### New APIs

- **`Kernel<'a, Args>`** — a typed handle that encodes a kernel's parameter signature in the type system. Wrong argument counts, types, or order are caught at compile time.
- **`KernelDescriptor`** trait — maps a kernel symbol name to its argument tuple. Provides a default `load(&Module) -> Kernel` implementation.
- **`#[derive(KernelDescriptor)]`** proc macro (`cust_derive`) — derive on a tuple struct with `#[kernel_name = "..."]` to auto-implement the trait.
- **`kernel_descriptor!`** declarative macro — function-signature syntax (`unsafe fn name(...)`) that generates a descriptor struct and `KernelDescriptor` impl. Supports optional `#[kernel_name = "..."]` override.
- **`typed_kernel!`** declarative macro — loads a kernel by name with an inline tuple type: `typed_kernel!(module, "name" => (T1, T2))`.
- **`Clone + Copy` for `Function` and `Kernel`** — enables cheap duplication and reuse across multiple streams.
- **Prelude re-exports** — `DevicePointer`, `DeviceBox`, `DeviceBuffer`, `DeviceCopy`, and `DeviceVariable` are now available via `cust::prelude::*`.
- **Occupancy helpers on `Kernel`** — forwards `suggested_launch_configuration`, `max_active_blocks_per_multiprocessor`, `available_dynamic_shared_memory_per_block`, and `get_attribute` from the underlying `Function`.
- **`KernelArgs` trait** — sealed unsafe trait implemented for tuples of `DeviceCopy` types. Expanded from 12 to 14 elements to support larger kernel signatures.

#### Example conversions

- `examples/gemm` — converted naive and tiled GEMM launchers to `KernelDescriptor::load` + `Kernel::launch`, with occupancy-driven block sizing via `kernel.as_function()`.
- `examples/gemv` — converted all ~30 kernel launch sites (f16, i8, Q4K, Q6K, ternary, MMA variants) to typed kernels.
- `samples/introduction/async_api` — converted the `increment` kernel launch to typed API.

#### Tests

- Added `trybuild` as a `dev-dependency` in `crates/cust/Cargo.toml`.
- Added trybuild UI tests for `#[derive(KernelDescriptor)]` in `crates/cust/tests/`:
  - 4 pass tests: unnamed struct, named struct, unit struct, single-field struct
  - 5 compile-fail tests: missing `#[kernel_name]`, wrong attribute type, generic struct, enum, union — with stable `.stderr` snapshots

#### Documentation

- New dedicated guide chapter: [Typed Kernels](guide/src/guide/typed_kernels.md) — covers loading, launch configuration, occupancy queries, multi-stream reuse, and raw-handle fallback.
- Updated [Getting Started](guide/src/guide/getting_started.md) with a "Typed kernel handles (recommended)" section.
- Updated [Kernel ABI](guide/src/guide/kernel_abi.md) with typed API examples after struct, slice, and reference sections.
- Updated [Safety](guide/src/guide/safety.md) to note that `Kernel::launch` eliminates param-count and type-order errors at compile time.
- Updated [Compute Capabilities](guide/src/guide/compute_capabilities.md) to recommend typed kernels across `#[cfg]` gated code paths.
- Updated [Tips](guide/src/guide/tips.md) with a tip recommending typed kernels over raw `launch!`.
- Updated [CUDA Pipeline](guide/src/cuda/pipeline.md) to mention `cust`'s typed kernel API.
- Added "Typed Kernel API (recommended)" section to the top-level [README](README.md).

## Earlier changes

See individual crate changelogs for pre-fork history:

- [`crates/cust/CHANGELOG.md`](crates/cust/CHANGELOG.md)
- [`crates/cuda_std/CHANGELOG.md`](crates/cuda_std/CHANGELOG.md)
- [`crates/rustc_codegen_nvvm/CHANGELOG.md`](crates/rustc_codegen_nvvm/CHANGELOG.md)
