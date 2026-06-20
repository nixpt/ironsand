---
type: howto
dcp: DCP/1.0
---

# LLVM 19 Build Recipe

The `rustc_codegen_nvvm` backend needs an LLVM toolchain whose major version
matches: **19** (with the `llvm19` cargo feature) or 7 (default). This box uses
LLVM 19. The official LLVM 19 release tarball (NOT from AUR — first-party
`github.com/llvm/llvm-project`) is extracted at `/workspace/scratch/llvm19`
(includes `llvm-config`, `llvm-as`, static libs, headers, `nvptx` component).

## Environment

```sh
export CUDA_PATH=/opt/cuda CUDA_ROOT=/opt/cuda CUDA_HOME=/opt/cuda
export LLVM_CONFIG_19=/workspace/scratch/llvm19/bin/llvm-config
export CARGO_TARGET_DIR=/workspace/scratch/builds/ironsand   # keep artifacts out of the repo subvol
```

## Build

```sh
# Codegen backend
cargo build -p rustc_codegen_nvvm --features llvm19

# Examples already bake the llvm19 feature into their cuda_builder build-dep,
# so they just work once the env above is set:
cargo build -p gemm -p vecadd -p matmul -p async_api

# Run (driver JIT-compiles the sm_100 PTX to the sm_120 GPU):
LD_LIBRARY_PATH=/opt/cuda/lib64 /workspace/scratch/builds/ironsand/debug/gemm
```

## Gotchas

- `cargo build --workspace` FAILS — the backend and `compiletests` default to
  the LLVM-7 path whose linux prebuilt is disabled. Build specific `-p` targets.
- Generated PTX is `.target sm_100` (Blackwell datacenter, the llvm19 default
  arch); it JITs forward-compat to the sm_120 5070 Ti.
- LLVM 22 is NOT a drop-in — the C++ shim is pinned to the LLVM 7/19 API. See
  the "LLVM 22 not viable" decision.
