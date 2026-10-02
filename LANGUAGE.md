# ironsand Language Specification

> Version: 0.1.0-draft  
> Scope: Consumer-facing API surface for downstream projects (zorro, zazen, and others).  
> haiku_san (the CPU/GPU orchestrator) lives outside ironsand and is out of scope.

This document defines the **ironsand language**: the vocabulary, contracts, and idioms that downstream consumers use to author GPU kernels in Rust and execute them via CUDA.

ironsand is organized into four layers. A typical consumer project uses all four:

1. **Build-time** (`cuda_builder`) — compile Rust kernel crates to PTX.
2. **Host runtime** (`cust`) — drive the GPU from CPU Rust.
3. **Device kernel** (`cuda_std`) — write GPU kernels in Rust.
4. **BLAS baseline** (`blastoff`) — cuBLAS bindings for reference GEMM.

---

## Table of Contents

- [1. Build-Time Language](#1-build-time-language)
- [2. Host-Side Runtime Language](#2-host-side-runtime-language)
- [3. Device-Side Kernel Language](#3-device-side-kernel-language)
- [4. Kernel ABI Contract](#4-kernel-abi-contract)
- [5. BLAS Baseline Language](#5-blas-baseline-language)
- [6. Compute Capability Gating](#6-compute-capability-gating)
- [7. Consumer Idioms](#7-consumer-idioms)
- [8. Glossary](#8-glossary)

---

## 1. Build-Time Language

Layer: `cuda_builder` crate (and underlying `nvvm`/`rustc_codegen_nvvm`).

### 1.1 Project Structure

Every ironsand consumer uses a **dual-crate layout**:

```
my_project/
├── Cargo.toml          # Host crate
├── build.rs            # Calls CudaBuilder
├── src/main.rs         # Host code
└── kernels/
    ├── Cargo.toml      # Kernel crate
    └── src/lib.rs      # #[kernel] functions
```

The kernel crate must declare:

```toml
[lib]
crate-type = ["cdylib", "rlib"]
```

- `cdylib` → compiled to PTX by `rustc_codegen_nvvm`.
- `rlib` → compiled normally so host code can import shared types.

### 1.2 CudaBuilder

`CudaBuilder` is the compiler driver invoked from `build.rs`:

```rust
use cuda_builder::CudaBuilder;

CudaBuilder::new(manifest_dir.join("kernels"))
    .arch(NvvmArch::Compute120)   // target GPU architecture
    .release(true)                // enable libnvvm optimizations
    .ftz(true)                    // flush denormals to zero
    .fma_contraction(true)        // enable fused multiply-add
    .override_libm(true)          // replace libm with libdevice intrinsics
    .copy_to(out_dir.join("kernels.ptx"))
    .build()
    .unwrap();
```

#### 1.2.1 Configuration Surface

| Field / Method | Type | Default | Description |
|----------------|------|---------|-------------|
| `arch` | `NvvmArch` | `Compute75` (legacy) / `Compute100` (llvm19) | Virtual compute architecture |
| `release` | `bool` | `true` | Release build; also sets `nvvm_opts = true` |
| `nvvm_opts` | `bool` | `true` if `release` | Run libnvvm optimizations |
| `generate_line_info` | `bool` | `true` | Emit debug line numbers |
| `debug` | `DebugInfo` | `None` | `None` or `LineTables` |
| `ftz` | `bool` | `false` | Flush single-precision denormals to zero |
| `fast_sqrt` | `bool` | `false` | Fast approximate `sqrt` |
| `fast_div` | `bool` | `false` | Fast approximate division |
| `fma_contraction` | `bool` | `true` | Enable FMA contraction |
| `override_libm` | `bool` | `true` | Redirect `libm` calls to libdevice |
| `use_constant_memory_space` | `bool` | `false` | Auto-place `static`s in constant memory |
| `optix` | `bool` | `false` | Aggressive inline + abort-on-panic (OptiX mode) |
| `emit` | `Option<EmitOption>` | `None` | Emit LLVM IR or bitcode for debugging |

#### 1.2.2 Architecture Targeting (`NvvmArch`)

`NvvmArch` selects the virtual compute architecture. Three suffix variants exist:

| Suffix | Meaning | Compatibility |
|--------|---------|---------------|
| *(none)* | Base | Forward-compatible to all future GPUs |
| `f` | Family-specific | Forward-compatible within same major version |
| `a` | Architecture-specific | Locked to exact compute capability |

```rust
NvvmArch::Compute120   // Base — forward compatible
NvvmArch::Compute120f  // Family — same major version only
NvvmArch::Compute120a  // Arch-specific — exact sm_120 only
```

The builder enables `#[cfg(target_feature = "compute_XXX")]` flags so kernel code can gate features:

```rust
#[cfg(target_feature = "compute_70")]
// Tensor cores available
```

### 1.3 Embedding PTX in the Host Binary

The host crate embeds the compiled PTX at compile time:

```rust
static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));
```

---

## 2. Host-Side Runtime Language

Layer: `cust` crate (safe CUDA Driver API).

### 2.1 Lifecycle

```rust
use cust::prelude::*;

// 1. Initialize CUDA driver and create a primary context
let _ctx = cust::quick_init()?;

// 2. Load PTX module (JIT-compiled by driver to device cubin)
let module = Module::from_ptx(PTX, &[])?;

// 3. Create an async stream
let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
```

### 2.2 Memory Vocabulary

All types require `T: DeviceCopy`. Derive it with `#[derive(DeviceCopy)]` (from `cust` or `cust_core`).

| Type | Location | Host Accessible | Typical Use |
|------|----------|-----------------|-------------|
| `DeviceBuffer<T>` | Device GPU | No | Bulk GPU data |
| `DeviceBox<T>` | Device GPU | No | Single GPU value |
| `DeviceSlice<T>` | Device GPU | No | View into device memory |
| `DevicePointer<T>` | Device GPU | No | Raw device pointer (FFI-safe) |
| `UnifiedBuffer<T>` | Unified | Yes | Simpler sharing; watch for page errors |
| `UnifiedBox<T>` | Unified | Yes | Single unified value |
| `LockedBuffer<T>` | Pinned host | Yes | Fast H↔D DMA |

**Traits:**
- `GpuBuffer<T>` — abstracts over `DeviceBuffer<T>` and `UnifiedBuffer<T>`.
- `GpuBox<T>` — abstracts over `DeviceBox<T>` and `UnifiedBox<T>`.
- `DeviceMemory` — raw pointer + size in bytes.

**Constructors:**

```rust
let buf = DeviceBuffer::from_slice(&[1.0f32; 1024])?;
let ptr = buf.as_device_ptr();
let len = buf.len();

// Copy H→D and D→H
buf.copy_from(&host_slice)?;
buf.copy_to(&mut host_slice)?;
```

### 2.3 Module & Function

```rust
// Load from PTX string, cubin bytes, fatbin bytes, or file path
let module = Module::from_ptx(PTX, &[])?;
let module = Module::from_cubin(&cubin_bytes, &[])?;
let module = Module::from_file("kernels.ptx")?;

// Retrieve kernel function by name
let kernel = module.get_function("gemv_warp")?;

// Query function attributes
let regs = kernel.get_attribute(FunctionAttribute::NumRegisters)?;
```

### 2.4 Launch Contract

Kernels are launched with the `launch!` macro using triple-chevron syntax:

```rust
unsafe {
    launch!(module.gemv_warp<<<grid, block, 0, stream>>>(
        a_gpu.as_device_ptr(),
        a_gpu.len(),
        b_gpu.as_device_ptr(),
        b_gpu.len(),
        c_gpu.as_device_ptr(),
        m, k, alpha, beta
    ))?;
}
```

**Launch parameters:**
- `grid` — `GridSize` (or `u32`, `(u32, u32)`, `(u32, u32, u32)`, `glam::UVec3`, `vek::Vec3<u32>`)
- `block` — `BlockSize` (same `From` impls as `GridSize`)
- `shared_mem_bytes` — dynamic shared memory size (usually `0`)
- `stream` — a `Stream` variable (must be a local ident; paths do not work)

The `launch!` macro statically asserts all arguments implement `DeviceCopy`.

### 2.5 Stream & Event Primitives

```rust
let stream = Stream::new(StreamFlags::NON_BLOCKING, Some(-1))?; // high priority
stream.synchronize()?;
stream.wait_event(&event, StreamWaitEventFlags::DEFAULT)?;
stream.add_callback(Box::new(|status| { ... }))?;

let event = Event::new(EventFlags::DEFAULT)?;
event.record(&stream)?;
```

### 2.6 CUDA Graphs

```rust
use cust::graph::{Graph, GraphCreationFlags, KernelInvocation};
use cust::kernel_invocation;

let mut graph = Graph::new(GraphCreationFlags::NONE)?;
let invocation = kernel_invocation!(kernel<<<grid, block, 0, stream>>>(...))?;
let node = graph.add_kernel_node(invocation, &[])?;
```

### 2.7 Error Model

All fallible operations return `CudaResult<T>` (`Result<T, CudaError>`).
`DropResult<T>` is used for fallible destructors (e.g., `Stream::drop`).

---

## 3. Device-Side Kernel Language

Layer: `cuda_std` crate (GPU standard library).

### 3.1 Kernel Definition

A kernel is an `unsafe extern "C"` function annotated with `#[kernel]`:

```rust
#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]

use cuda_std::prelude::*;

#[kernel]
#[allow(improper_ctypes_definitions)]
pub unsafe fn gemv_warp(
    a: &[f32],       // immutable slice → (ptr, len) pair
    x: &[f32],
    y: *mut f32,     // mutable output via raw pointer
    m: usize,
    k: usize,
) {
    // ...
}
```

The `#[kernel]` macro enforces:
- `extern "C"` calling convention
- `unsafe` function qualifier
- All parameters are `Copy`
- No return value
- `no_mangle` for stable symbol names

### 3.2 Thread Identity & Indexing

```rust
use cuda_std::thread;

let tx = thread::thread_idx_x();   // 0 .. block_dim_x - 1
let bx = thread::block_idx_x();    // 0 .. grid_dim_x - 1
let bdx = thread::block_dim_x();
let gdx = thread::grid_dim_x();

let global_1d = thread::index_1d();      // linear global thread index
let global_2d = thread::index_2d();      // UVec2
let global_3d = thread::index_3d();      // UVec3
let flat = thread::index();              // alias for index_1d

let ws = thread::warp_size();            // 32 on all current NVIDIA hardware
```

### 3.3 Synchronization & Fences

```rust
thread::sync_threads();                  // __syncthreads()
thread::sync_threads_count(pred);        // __syncthreads_count()
thread::sync_threads_and(pred);          // __syncthreads_and()
thread::sync_threads_or(pred);           // __syncthreads_or()
thread::grid_fence();                    // grid-level mem fence
thread::device_fence();                  // device-level mem fence
thread::system_fence();                  // system-level mem fence
thread::nanosleep(ns);                   // delay
```

### 3.4 Warp Primitives

```rust
use cuda_std::warp;

let lane = warp::lane_id();
let mask = warp::activemask();

// Shuffle
let (bits, _) = warp::warp_shuffle_xor(u32::MAX, value.to_bits(), offset, 32);

// Reductions (sum, min, max, etc.)
let sum = warp::warp_reduce_sum(u32::MAX, partial);

// Traits: WarpReduceValue, WarpShuffleValue, WarpMatchValue
```

### 3.5 Shared Memory

```rust
use cuda_std::shared;

let smem = shared::dynamic_shared_mem::<f32>();  // extern __shared__
```

### 3.6 Math & Types

**Half-precision floats:**

```rust
use cuda_std::f16;
use cuda_std::bf16;
```

**Math libraries:**
- `cuda_std::intrinsics` — raw libdevice bindings (`exp`, `log`, `sin`, `fma`, etc.)
- `glam` / `vek` — vector math types (re-exported)
- `half` crate — `f16`/`bf16` operations (re-exported)

**GPU-specific traits:**
- `GpuFloat` — float abstraction for generic kernels
- `FloatExt` — additional float methods

### 3.7 Inline PTX (`asm!`)

When LLVM/NVVM cannot express an instruction, use `asm!`:

```rust
core::arch::asm!(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32",
    "{{{d0,d1,d2,d3}}},",
    "{{{a0,a1}}},",
    "{{{b0,b1}}},",
    "{{{d0,d1,d2,d3}}};",
    // ... operands
);
```

Requires `#![cfg_attr(target_os = "cuda", feature(asm_experimental_arch))]`.

### 3.8 I/O & Debugging

```rust
cuda_std::println!("thread {} got value {}", thread::thread_idx_x(), x);
cuda_std::assert_eq!(actual, expected);
cuda_std::assert_ne!(actual, expected);
```

Output is buffered in a 1 MB circular buffer and flushed at synchronization points.

### 3.9 Atomics

```rust
use cuda_std::atomic;
```

See `atomic::intrinsics` for raw operations and `atomic::mid` for mid-level wrappers.

### 3.10 Panic & Allocation

- Panic calls `__nvvm_trap()` (immediate abort).
- `alloc` is available; allocation failure traps.
- The crate is `no_std` + `extern crate alloc`.

---

## 4. Kernel ABI Contract

This section defines how parameters cross the CPU→GPU boundary. It applies to `extern "C"` kernels (the only kind `#[kernel]` allows).

### 4.1 By-Value Passing

**Structs** and **arrays** are passed directly as byte arrays:

```rust
#[repr(C)]
pub struct Foo { a: u16, b: u64, c: u128 }

#[kernel]
pub unsafe fn kernel(foo: Foo) { }
```

→ PTX: `.param .align 16 .b8 kernel_param_0[32]`

Pass by value from host:

```rust
unsafe {
    launch!(module.kernel<<<1, 1, 0, stream>>>(foo))?;
}
```

### 4.2 Slices

Slices are passed as **two word-sized parameters**: `(ptr, len)`.

```rust
#[kernel]
pub unsafe fn kernel(a: &[u8]) { }
```

→ PTX: `.param .u64 kernel_param_0, .param .u64 kernel_param_1`

Host call:

```rust
unsafe {
    launch!(module.kernel<<<1, 1, 0, stream>>>(buf.as_device_ptr(), buf.len()))?;
}
```

**Mutable slices are disallowed** as kernel arguments (aliasing violation). Use `*mut T` or `&[UnsafeCell<T>]`.

### 4.3 Primitives

Passed directly by value. Map to PTX types `.s8`, `.s16`, `.s32`, `.s64`, `.u8`, `.u16`, `.u32`, `.u64`, `.f32`, `.f64`.

`u128`/`i128` are passed as byte arrays.

### 4.4 References & Pointers

Both passed as pointers. Host passes `DevicePointer<T>` or raw device addresses.

### 4.5 ZSTs

Zero-sized types are elided entirely.

### 4.6 repr(Rust) Warning

`repr(Rust)` types are discouraged in kernel parameters. Use `repr(C)` for stable layouts.

---

## 5. BLAS Baseline Language

Layer: `blastoff` crate (cuBLAS bindings).

Used for baseline/reference GEMM and vector operations when hand-written kernels are not required.

```rust
use blastoff::CublasContext;

let mut cublas = CublasContext::new()?;
cublas.set_math_mode(MathMode::DEFAULT | MathMode::TF32_TENSOR_OP)?;

cublas.with_stream(&stream, |ctx| {
    ctx.gemm(
        MatrixOp::None, MatrixOp::None,
        m, n, k,
        &alpha,
        a_ptr, lda,
        b_ptr, ldb,
        &beta,
        c_ptr, ldc,
    )
})?;
```

**Vocabulary:**
- `CublasContext` — cuBLAS handle (one per thread/device)
- `MathMode` — precision flags (`DEFAULT`, `PEDANTIC`, `TF32_TENSOR_OP`, `DISALLOW_REDUCED_PRECISION_REDUCTION`)
- `MatrixOp` — `None`, `Transpose`, `ConjugateTranspose`
- `GemmDatatype` — `f16`, `f32`, `f64`, `Complex32`, `Complex64`
- Level-1/Level-3 operations: `axpy`, `dot`, `gemm`, etc.

---

## 6. Compute Capability Gating

Kernel code can adapt to GPU features at compile time via `#[cfg(target_feature = ...)]`.

### 6.1 Target Features

The `CudaBuilder::arch()` choice enables a set of `target_feature` flags:

```rust
#[cfg(target_feature = "compute_70")]
// Code requiring compute 7.0+

#[cfg(target_feature = "compute_100")]
// Code requiring Blackwell base features

#[cfg(target_feature = "compute_120a")]
// Code locked to exact sm_120
```

### 6.2 Capability Hierarchy

- Base arch enables all lower base variants: `Compute120` enables `compute_50` through `compute_120`.
- Family variant (`f`) enables base + same-major family variants up to its minor version.
- Architecture variant (`a`) enables base + family + itself.

### 6.3 Common Patterns

```rust
// At least a capability
#[cfg(target_feature = "compute_60")]
{ /* f64 atomics */ }

// Exactly one capability
#[cfg(all(target_feature = "compute_61", not(target_feature = "compute_62")))]
{ /* 6.1-specific */ }

// Range
#[cfg(all(target_feature = "compute_70", not(target_feature = "compute_90")))]
{ /* Turing/Ampere only */ }
```

---

## 7. Consumer Idioms

### 7.1 The Dual-Crate Pattern

Host and kernel crates share types via the kernel crate's `rlib` compilation:

```rust
// kernels/src/lib.rs
pub type Elem = f32;

#[kernel]
pub unsafe fn add(a: &[Elem], b: &[Elem], c: *mut Elem) { }
```

```rust
// src/main.rs
use kernels::Elem;

let a: [Elem; 4] = [1.0, 2.0, 3.0, 4.0];
```

Changing `Elem` in one place updates both sides.

### 7.2 Raw Pointers for Mutable Outputs

Kernels take `&[T]` for inputs and `*mut T` for outputs. Never `&mut [T]`.

### 7.3 Warp-Centric Optimization

Optimized kernels think in warps (32 lanes), not threads:

```rust
let lane = thread::thread_idx_x() % 32;
let warp_id = thread::thread_idx_x() / 32;
```

Use warp shuffles for reductions instead of shared memory when possible.

### 7.4 Async by Default

All kernel launches are async. Always `stream.synchronize()` before reading GPU memory from CPU.

### 7.5 Escape Hatch: Inline PTX

When the compiler cannot generate an instruction (tensor-core `mma.sync`, `ldmatrix`, `bar.sync`), use `asm!`.

---

## 8. Glossary

| Term | Meaning |
|------|---------|
| **PTX** | Parallel Thread Execution — NVIDIA's intermediate ISA |
| **cubin** | Compiled GPU binary (architecture-specific) |
| **fatbin** | Container with multiple cubin/PTX variants |
| **NVVM** | NVIDIA's LLVM IR compiler (libnvvm) |
| **SM** | Streaming Multiprocessor |
| **Warp** | Group of 32 threads executing in SIMT lockstep |
| **Shared memory** | On-chip scratchpad per block (~48–228 KB) |
| **Constant memory** | Read-only cached device memory (~64 KB) |
| **Dynamic shared memory** | `extern __shared__` — sized at launch time |
| **DeviceCopy** | Trait marking types safe to copy across the host/device boundary |
| **Grid** | Collection of blocks launched together |
| **Block** | Collection of threads that can share memory and synchronize |
| **Stream** | Ordered sequence of GPU operations |
| **Event** | Synchronization primitive for cross-stream ordering |
| **Tensor core** | Mixed-precision matrix-multiply accumulator unit |
| **mma.sync** | Synchronous tensor-core matrix operation (inline PTX) |
| **WGMMA** | Warp-group matrix multiply accumulate (Blackwell) |

---

## Appendix: Crate Map

| Crate | Role | Consumer Uses It For |
|-------|------|----------------------|
| `cuda_builder` | Build driver | `build.rs` — compile kernels to PTX |
| `cust` | Host runtime | Memory, modules, streams, launches |
| `cust_core` | Shared types | `DeviceCopy` trait for cross-crate sharing |
| `cust_derive` | Derive macros | `#[derive(DeviceCopy)]` |
| `cuda_std` | Device stdlib | `#[kernel]`, thread/warp/math intrinsics |
| `nvvm` | NVVM wrapper | Direct libnvvm access (usually via `cuda_builder`) |
| `blastoff` | cuBLAS bindings | Baseline GEMM/BLAS |
| `gpu_rand` | GPU RNG | Random number generation in kernels |

---

*End of specification.*
