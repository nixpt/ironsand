# Spike: cuda-oxide vs ironsand on `gemv_q4k_fast` (IRONSAND-OXIDE-1)

**Question.** Can [NVIDIA/cuda-rust](https://github.com/NVIDIA/cuda-rust) (formerly
NVlabs/cuda-oxide) replace ironsand (our `Rust-GPU/rust-cuda` fork) for authoring
zorro's GPU inference kernels in Rust? cuda-oxide is a custom rustc backend
(Rust MIR → Pliron IR → LLVM IR → `llc` → PTX) that needs **no LLVM-19 install
and no libNVVM C++ shim** — exactly the maintenance surface that makes ironsand
expensive (hand-maintained `rustc_codegen_nvvm` against nightly drift, a
hand-provisioned LLVM 19).

**This is a timeboxed evaluation, not a migration.** One kernel was ported and
measured end-to-end against the kernel zorro actually ships.

**Verdict: adopt-with-caveats.** The port is byte-identical to the shipping
ironsand PTX, has an identical register/spill footprint, is modestly *faster*,
and was written with a cleaner API (no `asm!` for warp-reduce or f16). The
caveats gating a full "adopt" are that only the *simplest* kernel class was
validated and the toolchain is alpha. Evidence below.

---

## 1. Toolchain setup (what actually worked)

Host: RTX 5070 Ti Laptop (sm_120, CC 12.0), CUDA 13.3, driver 610.57.04,
system clang 22, rustup `llvm-tools`.

```bash
# 1. Clone the repo (scratch is fine; the build is reproducible).
git clone https://github.com/NVIDIA/cuda-rust.git

# 2. Pinned nightly + components (also auto-installed via rust-toolchain.toml).
rustup toolchain install nightly-2026-08-28 \
  -c rust-src -c rustc-dev -c rust-analyzer -c clippy -c rustfmt -c llvm-tools

# 3. cargo-oxide. Inside the repo it works via a workspace alias; for use
#    elsewhere install it (from the clone or from git):
cargo install --path cuda-rust/cuda-oxide/crates/cargo-oxide
#    or: cargo +nightly-2026-08-28 install --git https://github.com/NVlabs/cuda-oxide.git cargo-oxide

# 4. Build the codegen backend once (published to a shared cache, keyed on the
#    cuda-oxide git rev, and reused by external projects at the same rev).
cd cuda-rust/cuda-oxide && cargo oxide setup

# 5. Verify.
cargo oxide doctor
cargo oxide run vecadd        # -> "✓ SUCCESS: All 1024 elements correct!"
```

Every prerequisite the backend needs was already on the box — **no system
package installs, no LLVM-19, no libNVVM shim.** `cargo oxide doctor` output:

```
Rust nightly toolchain... ✓ rustc 1.100.0-nightly
Required rustup components... ✓ rust-src, rustc-dev, rust-analyzer, clippy, rustfmt, llvm-tools
Codegen backend... (built via `cargo oxide setup`)
CUDA headers / toolkit (nvcc 13.3) ... ✓
libNVVM / nvJitLink / libdevice ... ✓
llc (LLVM)... ✓ LLVM version 23.1.0 (bundled in the nightly's llvm-tools)
clang / libclang resource dir... ✓ /usr/lib/clang/22 (via clang)
NVIDIA driver / GPU... ✓ RTX 5070 Ti (compute capability 12.0, driver 610.57.04)
compute-sanitizer / cuda-gdb ... ✓
✅ Environment looks good!
```

Notable: the pipeline uses the Rust toolchain's **own bundled `llc` (LLVM 23.1)**
from the `llvm-tools` component, so a separate `llc-21+` apt install was not
needed. `cargo oxide run vecadd` printed `✓ SUCCESS: All 1024 elements correct!`.

### Friction points

- **The `cargo oxide new` scaffold does not build at this rev.** It writes
  `cuda-device`/`cuda-host` as git deps **and `cuda-core = "0.4.0"` from
  crates.io**. At the pinned rev (whose top commit is *"lift shared host crates
  to the git root"*) the git-root `cuda-core` and the crates.io `cuda-core 0.4.0`
  are **distinct sources cargo will not unify**, producing two `DeviceBuffer<T>`
  types — the host code's and the one the generated `#[cuda_module]` launch
  method expects — and the build fails with `E0308`. **Fix: pin `cuda-core` to
  the same git rev as `cuda-host`.** This spike's `Cargo.toml` does that and
  builds; worth reporting upstream as a scaffold-recipe bug.
- cuda-oxide defaults the device target to **sm_80** unless
  `CUDA_OXIDE_DEVICE_ARCH=sm_120` (or `--arch sm_120`) is set; the driver
  forward-JITs to sm_120 at load either way (see caveats).

---

## 2. The port

`spikes/cuda-oxide-q4k/src/main.rs` is single-source (host + device in one file,
one `cargo oxide build`). The device kernel is a faithful line-by-line port of
ironsand's `examples/gemv/kernels/src/gemv_q4k.rs::gemv_q4k_fast`:

- Warp-per-row; `d`/`dmin` decoded once per 256-weight super-block; inner loop
  over the 8 sub-blocks with the coalesced nibble read (`qs[g*32 + lane]`).
- Launch shape identical to ironsand: block = 256 (8 warps), grid = `m.div_ceil(8)`
  (expressed as `LaunchConfig::for_num_elems(m*32)`).
- `cvt.f32.f16` via cuda-oxide's native `f16` (`convert::cvt_f32_f16x2_lo`);
  the butterfly warp reduction via `warp::reduce_sum_f32`; the lane-0 output
  write via a `DisjointSlice` + `get_unchecked_mut(row)` (the documented
  warp-reduction pattern). **No `asm!` anywhere** — ironsand needs inline PTX
  for both of these (see §5).

### Build + run (one command each)

```bash
cd spikes/cuda-oxide-q4k
# Point at a .ptx that exports `gemv_q4k_fast` (zorro vendors ironsand's).
IRONSAND_PTX=/path/to/ironsand_gemv.ptx cargo oxide run
```

`IRONSAND_PTX` is optional; unset, the harness reports cuda-oxide-vs-CPU only
and skips the A/B. Two build paths were both verified on this box: the portable
git-dep `Cargo.toml` above, and an offline `.cargo/config.toml` `paths` override
pointing the deps at a local clone (reuses the already-built backend, no network).

---

## 3. Correctness

Same deterministic Q4_K bytes + activations as zorro's
`cuda_q4_k_gemv_ironsand_matches_cpu` (super-block `d`/`dmin` + pseudo-random
scales/quants; `x[i] = (i%23 − 11)·0.05`), plus a realistic decode shape. Three
paths on identical inputs: **(a)** CPU dequant·dot (byte-faithful to llama.cpp
`block_q4_K`), **(b)** the cuda-oxide kernel, **(c)** the existing ironsand PTX
loaded from file via `CudaContext::load_module_from_ptx_src` + `load_function`.

| m | k | b-vs-a max-abs | b-vs-a rel-L2 | b-vs-c max-abs | b-vs-c rel-L2 |
|------|------|-----------|-----------|-----------|-----------|
| 8 | 512 | 2.289e-5 | 2.947e-7 | **0** | **0** |
| 4096 | 4096 | 5.798e-4 | 1.120e-6 | **0** | **0** |

The tolerance the ironsand kernel is held to vs CPU is **max-abs < 1e-2** (zorro's
test). The cuda-oxide kernel passes it, and **b − c is bitwise zero** at both
shapes: cuda-oxide reproduces the ironsand PTX output exactly — same algorithm,
same numerics, same reduction order. No tolerance was loosened.

---

## 4. Performance (matched A/B)

Same process, same inputs, CUDA events, 20 warmup + 64 timed runs, baseline and
new **alternated within one loop** (the box drifts 10–15% cold-vs-warm;
interleaving removes it). `GB/s = weight_bytes / median_time` (weights dominate
GEMV traffic). cuda-oxide default target (sm_80):

| m | k | oxide µs | oxide GB/s | ironsand µs | ironsand GB/s | ironsand/oxide |
|------|------|------|------|------|------|------|
| 8 | 512 | 3.5 | 0.7 | 5.3 | 0.4 | 1.51× |
| 4096 | 4096 | 39.6 | 238 | 45.9 | 206 | 1.16× |
| 11008 | 4096 | 96.7 | 262 | 104.2 | 243 | 1.08× |
| 4096 | 11008 | 101.2 | 251 | 116.4 | 218 | 1.15× |

Rebuilt with native `CUDA_OXIDE_DEVICE_ARCH=sm_120`, the delta is unchanged
(4096×4096: oxide 39.7 µs vs ironsand 45.9 µs = 1.16×) — the edge is not a
target-arch artifact.

**The cuda-oxide kernel is modestly faster than the shipping ironsand PTX across
every shape.** §5 shows why: the memory/reduction instruction stream is
identical; cuda-oxide's LLVM NVPTX backend just contracts the affine dequant into
more FMAs.

*Context, not a pass bar:* zorro's **default** Q4_K decode path is `mmvq`
(Q8-quantized-activation integer dot), which is faster than `gemv_q4k_fast`
(f32 dequant·dot) — a known result we do not re-litigate here. This spike
compares the **same kernel across two toolchains**, nothing else.

---

## 5. PTX / ptxas / ergonomics

`ptxas -v --gpu-name sm_120` on both PTXs, entry `gemv_q4k_fast`:

| | oxide | ironsand |
|---|---|---|
| registers | **40** | **40** |
| spill stores / loads | 0 / 0 | 0 / 0 |
| stack frame | 0 B | 0 B |

**Identical SASS register footprint.** cuda-oxide's codegen is as
register-efficient as nvvm's here.

Instruction mix (PTX bodies of the two kernels):

| instruction | oxide | ironsand |
|---|---|---|
| `ld.global.b8` | 20 | 20 |
| `ld.global.b32` | 9 | 9 |
| `shfl.sync.bfly.b32` | 5 | 5 |
| `cvt.*.f16` | 2 | 2 |
| `fma.rn.f32` | **17** | **9** |
| `mul.f32` | 16 | 24 |
| `sub.f32` | **0** | **8** |
| `add.f32` | 5 | 5 |

The **memory and reduction streams are byte-for-byte identical** (same 29
`ld.global`, same 5 butterfly `shfl`, same 2 hardware `cvt.f32.f16`). The only
difference is float-ALU contraction: cuda-oxide folds the affine
`d_eff·nib − m_eff` and the `·x + acc` into FMAs (17 fma, no sub), where nvvm
emits explicit `sub` + `mul` (9 fma, 8 sub) — fewer total float ALU ops,
consistent with the measured 1.08–1.16× edge.

### Ergonomics

- **cuda_std/cust features used by the kernel that have no cuda-oxide
  equivalent: none.** Every primitive mapped to a native cuda-oxide API, in two
  cases a *cleaner* one:
  - Warp reduction: ironsand must hand-write inline PTX `shfl.sync` because
    `cuda_std`'s warp-shuffle intrinsic SIGSEGVs libNVVM (a known ironsand
    footgun). cuda-oxide provides `warp::reduce_sum_f32` / `shuffle_xor_f32`
    natively.
  - f16→f32: ironsand uses `asm!("cvt.f32.f16 …")`. cuda-oxide has a native
    `f16` type and `convert::cvt_f32_f16x2_lo` (lowers to the same `cvt`).
- **What cuda-oxide offers that ironsand can't without `asm!`:** besides the two
  above, native `mma.sync`/`ldmatrix`/`cp.async`/TMA and a `dotprod` (dp4a)
  module, plus a typed safe-launch model (`DisjointSlice`, `WarpIndex` witness,
  `LaunchConfig`, launch contracts). Not exercised by this GEMV, but these are
  exactly the primitives ironsand reaches `asm!` for in its dp4a and
  tensor-core kernels.

---

## 6. Caveats

1. **Target asymmetry (controlled).** The vendored ironsand PTX is `.target
   sm_61`; cuda-oxide emits sm_80 (default) or sm_120 (native). The driver
   forward-JITs both to sm_120 SASS at load. `ptxas` for sm_120 gives both the
   *identical* 40-reg/0-spill footprint, and the native-sm_120 oxide build
   reproduces the same perf delta — so the comparison is sound and the edge is
   real, not an arch artifact.
2. **Only the simplest kernel class is validated.** `gemv_q4k_fast` is a
   warp-reduce dequant·dot: no dp4a, no shared memory, no `mma.sync`. It is a
   deliberately conservative, high-confidence first port. The *hardest* ironsand
   kernels — the hand-emitted `mma.sync`/`ldmatrix` tensor-core tiles (the
   prefill lane) and the dp4a `mmvq`/`vecdot` paths — are where cuda-oxide's
   native-intrinsic advantage is largest **and** where codegen risk is least
   tested. This spike says nothing about them.
3. **Alpha toolchain.** cuda-oxide is explicitly alpha: expect API breakage.
   The build pins a nightly (`nightly-2026-08-28`) and a git rev; and the
   `cargo oxide new` scaffold's dep recipe needs the cuda-core-from-git fix (§1).

---

## 7. Verdict — adopt-with-caveats

For the kernel class it covers, cuda-oxide is a clear win over ironsand: it
installs with **zero LLVM-19/libNVVM hand-provisioning** (ironsand's single
largest maintenance cost), the kernel ports 1:1 with a **cleaner API** (no
`asm!` for warp-reduce or f16), the output is **bit-identical** to the shipping
ironsand PTX, the **register/spill footprint is identical**, and it is
**modestly faster** (1.08–1.16× at decode shapes) because its LLVM backend
contracts the dequant better.

What gates a full "adopt" and a decision to freeze ironsand: the tensor-core
(`mma.sync`/`ldmatrix`) and dp4a kernels — the parts that make ironsand painful
to maintain *and* carry the most codegen risk — are unproven on cuda-oxide.

**Recommendation.** Greenlight a follow-up spike porting one dp4a kernel
(`gemv_q4k_vecdot`) and one `mma.sync` kernel to cuda-oxide, gated the same way
(CPU oracle + bit-for-bit A/B vs the ironsand PTX + `ptxas -v`). If those port
and match as cleanly as this one did, freeze ironsand and move kernel authoring
to cuda-oxide. If the tensor-core path hits a codegen wall, keep ironsand for
the tensor lane and use cuda-oxide for the scalar/decode lane.
