# Attribution

`ironsand` is a **hard fork** of [The Rust CUDA Project](https://github.com/Rust-GPU/rust-cuda)
(`Rust-GPU/rust-cuda`), which is itself a reboot of the original `rust-cuda` started by
Riccardo D'Ambrosio and the Rust GPU community.

The overwhelming majority of the code in this repository — the `rustc_codegen_nvvm` codegen
backend, the `cust` host-side stack, `cuda_std`, `cuda_builder`, `nvvm`, the PTX tooling, and
the examples — originates from that upstream project and remains under its original authors'
copyright.

- Upstream: https://github.com/Rust-GPU/rust-cuda
- Original `rustc_codegen_nvvm` copyright: Copyright (c) 2021 Riccardo D'Ambrosio
- Licensed under MIT OR Apache-2.0; see [LICENSE-MIT](LICENSE-MIT) and
  [LICENSE-APACHE](LICENSE-APACHE). Those notices are preserved unchanged.

## How ironsand differs in purpose

Upstream Rust CUDA aims to be a **general-purpose** ecosystem for writing and running arbitrary
GPU code in Rust (compute, ray tracing via OptiX, deep-learning primitives via cuDNN, etc.).

`ironsand` has a **narrower, different purpose**: it is an experimentation vehicle for the
[`zorro`](../../projects/zorro) inference engine — specifically, exploring authoring GPU kernels
for LLM inference (GEMM, attention, sampling, etc.) in Rust via the Rust→PTX path, as an
alternative to hand-written CUDA C++.

Because of that narrower goal, this fork has dropped upstream components that are not relevant to
inference experimentation:

- **OptiX** (ray tracing) — `optix*` crates and the path-tracer example.
- **cuDNN** (`cudnn`, `cudnn-sys`) — `zorro` uses its own kernels rather than cuDNN.
- Non-inference examples (`i128_demo`, `sha2_crates_io`) and the `async_api` sample.
- Upstream CI / packaging infrastructure (`container/`, `.github/`, Nix flake, devcontainer,
  OptiX/vast.ai helper scripts).

These removals are a scoping choice for this fork, not a judgement on upstream. Anything dropped
can be recovered from the upstream repository.
