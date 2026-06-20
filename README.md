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
