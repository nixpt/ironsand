# The CUDA pipeline

CUDA is traditionally used via CUDA C/C++ files which have a `.cu` extension. These files can be
compiled using NVCC (NVIDIA CUDA Compiler) into an executable.

CUDA files consist of **device** and **host** functions. **Device** functions run on the GPU, and
are also called kernels. **Host** functions run on the CPU and usually include logic on how to
allocate GPU memory and call device functions.

Behind the scenes, NVCC has several stages of compilation.

First, NVCC separates device and host functions and compiles them separately. Device functions are
compiled to [NVVM IR](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html), a subset of LLVM IR
with additional restrictions including the following.
- Many intrinsics are unsupported.
- "Irregular" integer types such as `i4` or `i111` are unsupported and will segfault (however in
  theory they should be supported).
- Global names cannot include `.`.
- Some linkage types are not supported.
- Function ABIs are ignored; everything uses the PTX calling convention.

libNVVM is a closed source library which takes NVVM IR, optimizes it further, then converts it to
PTX. PTX is a low level, assembly-like format with an open specification which can be targeted by
any language. For an assembly format, PTX is fairly user-friendly.
- It is well formatted.
- It is mostly fully specified (other than the iffy grammar specification).
- It uses named registers/parameters.
- It uses virtual registers. (Because GPUs have thousands of registers, listing all of them out
  would be unrealistic.)
- It uses ASCII as a file encoding.

PTX can be run on NVIDIA GPUs using the driver API or runtime API. Those APIs will convert the PTX
into a final format called SASS which is register allocated and executed on the GPU.

## The ironsand pipeline

The ironsand project replaces NVCC with a custom rustc backend. The pipeline looks like this:

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                        The ironsand pipeline                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Host compile-time                    Device compile-time                  │
│   ────────────────                     ──────────────────                   │
│                                                                             │
│        ┌──────────────┐                      ┌──────────────┐               │
│        │ Host Rust    │                      │ GPU kernel   │               │
│        │ code (.rs)   │                      │ Rust (.rs)   │               │
│        └──────┬───────┘                      └──────┬───────┘               │
│               │                                     │                       │
│               │ rustc (host backend)                │ rustc_codegen_nvvm    │
│               │                                     │ (custom backend)      │
│               ▼                                     ▼                       │
│        ┌──────────────┐                      ┌──────────────┐               │
│        │ Host binary  │<---- include_str!()─│ PTX (.ptx)   │               │
│        │ (ELF)        │      embeds PTX      │              │               │
│        └──────┬───────┘                      └──────┬───────┘               │
│               │                                     │                       │
│               │                                     ▲                       │
│               │                              ┌──────┴──────┐                │
│               │                              │ libNVVM     │                │
│               │                              │ (NVVM IR    │                │
│               │                              │  --> PTX)   │                │
│               │                              └─────────────┘                │
│               │                                                             │
│   Runtime     │                                                             │
│   ───────     ▼                                                             │
│        ┌──────────────┐                                                     │
│        │ cust         │  Rust safe wrapper around CUDA Driver API           │
│        │ (Rust)       │  ├─ Module::from_ptx(PTX)                           │
│        └──────┬───────┘  ├─ Kernel<'a, Args>::launch(...)                   │
│               │          └─ Stream, Event, DeviceBuffer                     │
│               │                                                             │
│               ▼                                                             │
│        ┌──────────────┐                                                     │
│        │ CUDA Driver  │  cuModuleLoad, cuLaunchKernel, cuStreamSynchronize   │
│        │ API (C)      │                                                     │
│        └──────┬───────┘                                                     │
│               │                                                             │
│               ▼                                                             │
│        ┌──────────────┐                                                     │
│        │ JIT compile  │  Driver compiles PTX → SASS (machine code)          │
│        │ (PTX → SASS) │  at load time for the target GPU architecture        │
│        └──────┬───────┘                                                     │
│               │                                                             │
│               ▼                                                             │
│        ┌──────────────┐                                                     │
│        │ GPU Execution│  Warps, threads, shared memory, tensor cores        │
│        │ (SASS)       │                                                     │
│        └──────────────┘                                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

- **`rustc_codegen_nvvm`** is a custom rustc backend that compiles GPU kernel crates to NVVM IR
  (LLVM bitcode) instead of the usual host target.
- **`cuda_std`** provides the GPU-side standard library (thread indexing, shared memory,
  intrinsics, etc.) used inside kernel crates.
- **`cuda_builder`** is a build-script helper that drives `rustc_codegen_nvvm` from a host
  crate's `build.rs`, producing a `.ptx` file that is embedded in the host binary.
- **`cust`** is the host-side safe wrapper around the CUDA Driver API, used to load modules,
  allocate GPU memory, launch kernels, and synchronize results. `cust` also provides the typed
  [`Kernel`](../../kernel/struct.Kernel.html) API, which encodes a kernel's parameter signature
  in the Rust type system for compile-time launch safety (see [Typed Kernels](../guide/typed_kernels.html)).