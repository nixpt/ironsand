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

## Provisioning (s472) — do not hand-extract

```sh
scripts/provision-llvm19.sh          # idempotent: fetch, verify, extract, self-check
scripts/provision-llvm19.sh --check  # verify only; loud non-zero exit if unusable
```

Pinned artifact, verified against upstream's **SLSA/sigstore attestation**
(`LLVM-19.1.7-Linux-X64.tar.xz.jsonl`, subject digest matched, builder
`github-hosted actions runner`) — not merely self-consistent with whatever was
downloaded:

| | |
|---|---|
| version | 19.1.7 |
| size | 1,653,440,720 B (1.65 GB) → ~8.2 GB extracted |
| sha256 | `4a5ec53951a584ed36f80240f6fbf8fdd46b4cf6c7ee87cc2d5018dc37caf679` |

**Why a script and not just this prose:** the prose already existed and the
toolchain still vanished. `/workspace/scratch` is explicitly disposable
("`rm -rf scratch/*` must always be safe"), so an 8.2 GB hand-extraction there
is a capability with no restore path and no absence check. It was reclaimed,
and `cargo build -p rustc_codegen_nvvm --features llvm19` failed with
*"no LLVM 19 toolchain was found"* for weeks before anyone looked (s472).
The script makes restore one command and makes absence loud.

**Why we do NOT fork llvm-project:** we carry zero patches to LLVM. A fork's
job is to hold patches (cf. `nixpt/llama.cpp` branch `nixpt-oracle`, which
exists because we genuinely patch llama.cpp). Here we consume a prebuilt
release artifact — forking the source would not even produce the binaries we
link against, only an obligation to build LLVM ourselves. Pin the artifact,
not the source.

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
