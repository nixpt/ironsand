---
type: howto
dcp: DCP/1.0
status: completed-migration  (Steps 1-3 of 4 landed; Step 4 separate track — see Open followups)
renamed-from: llvm20-runtime-shim-recipe.md
phase-3d-update: |-
  2026-06-29 — Steps 1-3 cfg-gates were REVERSED in commit 79899f8 (`phase-3d: drop llvm20 + llvm22 plumbing`). The 5x `cfg(not(any(feature=llvm19, llvm20, llvm22)))` legacy arms collapsed to `cfg(not(feature=llvm19))`; the 3x `cfg(any(feature=llvm20, llvm22))` post-LLVM-17 shim arms (incl. `LLVMRustFindAndCreatePass` rehydration in `dce_pass`, the `LLVMBuildLoad2` typed-load bridge, and `LLVMRustConstZExt`) were DELETED.
  Only Step 4 (the `LLVMRustStringWriteImpl` FFI body at `src/init.rs:140-185`) remains current, since the `RustString` struct + unsafe-extern body are still required to keep the cdylib dlopen-clean under `cargo -Zcodegen-backend` even on the llvm19 path.
  Cross-link update: the in-body Cross-links entry that previously pointed at `.dejavue/references/llvm22-build-recipe.md` was patched to point at `.dejavue/references/llvm19-llvm22-migration-recipe-archived.md` (the post-Phase-3d rename target).
---

# LLVM 20 Runtime-Shim Recipe — Landed Implementation

Steps 1-3 of the original four-symbol runtime shim are landed and
verified clean across both `--features llvm19` and `--features llvm20`
(`cargo check -p rustc_codegen_nvvm --features llvm19` and `--no-default-
features --features llvm20` both RC=0, no new diagnostics introduced
by this set of edits). Step 4 (`LLVMRustStringWriteImpl`) lives on a
separate track documented in `.dejavue/decisions.md` (see cross-links
below).

> **Note on line numbers.** The references below are the actual
> landed line numbers in the post-style-collapse state of the crates.
> A few are slightly offset from the pre-edit spec the recipe was
> originally drafted against (`src/nvvm.rs:343` (pre-edit) landed at
> `src/nvvm.rs:375-385` (post-style-collapse); `src/llvm.rs:1587`
> (pre-edit) landed at `src/llvm.rs:1595-1598`). In all cases the
> symbol-level intent is unchanged; if you grep the original spec
> number, find the nearest landed citation to confirm. If a future
> edit grows any referenced region by more than a handful of lines,
> update the citation here in the same change.

## Landed surface area (concrete file/line references)

### Step 1 — `LLVMBuildLoad` (untyped) → typed `LLVMBuildLoad2`

- Legacy untyped FFI decl: `src/llvm.rs:1922` — gated
  `#[cfg(not(any(feature = "llvm19", feature = "llvm20", feature = "llvm22")))]`
  so the symbol disappears from the cdylib under all three non-7
  toolchains.
- `LLVMBuildLoad2` typed decl (already unconditionally exported) at
  `src/llvm.rs:~1910-1920` — used by the four call sites in
  `src/builder.rs:515, 528, 1234, 1299`. Each site is cfg-dispatched:
  the LLVM 19/20/22 arm calls `LLVMBuildLoad2(B, val_ty, ptr, UNNAMED)`;
  the LLVM 7 arm keeps the untyped form.

### Step 2 — `LLVMConstZExt` → `LLVMRustConstZExt` shim

- Legacy `LLVMConstZExt` FFI decl: `src/llvm.rs:1595-1596` — gated
  `#[cfg(not(any(feature = "llvm19", feature = "llvm20", feature = "llvm22")))]`
  so it survives only on the LLVM 7 default-features path.
- New Rust-side `LLVMRustConstZExt` decl: `src/llvm.rs:1597-1598` —
  gated `#[cfg(any(feature = "llvm20", feature = "llvm22"))]`.
- C++ shim body: `RustWrapper.cpp:2272-2278` (collapsed header
  comment, 3-line breadcrumb). Uses
  `ConstantExpr::getCast(Instruction::ZExt, …)` because
  `ConstantExpr::getZExt` is not public even on LLVM 19.
- Call site: `src/consts.rs:361-373`. The body comment is at lines
  361-364, the `llvm20/llvm22` arm invoking `llvm::LLVMRustConstZExt`
  at line 367, and the legacy `llvm7` arm keeping the original
  `llvm::LLVMConstZExt` at line 373.

### Step 3 — `LLVMAddGlobalDCEPass` → name-registry route

- Legacy PM FFI decl: `src/llvm.rs:~976-982` — gated
  `#[cfg(not(any(feature = "llvm19", feature = "llvm20", feature = "llvm22")))]`
  so the LLVM 19/20/22 builds don't even mention it. Header comment
  collapsed to a 5-line breadcrumb above the gate.
- New arm in `dce_pass`: `src/nvvm.rs:375-385`. The legacy-PM pass
  is rehydrated via `LLVMRustFindAndCreatePass(c"globaldce".as_ptr().cast(), 9)`
  and fed to the legacy pass manager via `LLVMRustAddPass`. A
  sentinel `const _: usize = b"globaldce".len();` at `src/nvvm.rs:378`
  keeps the byte literal referenced so a future typo cannot silently
  drift from the explicit length arg `9` passed to the FFI.

### Step 1 cross-link to LLVM 19 tightening

The cfg-gate tightening that closed Step 1 also positively impacts
the LLVM 19 production path: the previous gates
(`#[cfg(not(any(feature = "llvm20", feature = "llvm22")))]`) left
dangling FFI decls under `--features llvm19` that the linker was
silently eliding. The post-recipe gates
(`not(any(feature = "llvm19", feature = "llvm20", feature = "llvm22"))`)
make the LLVM 19 path's intent match the post-LLVM-17 reality.

## Toolchain contract (verified)

LLVM 20.1.8 from `release/20.x` is at `/workspace/scratch/llvm20`
(cmake + ninja, NVPTX + X86 only). `nm -D libLLVM-20.so` confirms:

- **Removed** (no longer exported): `LLVMBuildLoad`, `LLVMConstZExt`,
  `LLVMAddGlobalDCEPass`.
- **Replacement symbols present** in `libLLVM-20.so`: `LLVMBuildLoad2`
  (typed builder), `LLVMConstInt` (const-int API family), and the
  new-PM pass-name registry that backs `LLVMRustFindAndCreatePass`.

## Environment

```sh
export CUDA_PATH=/opt/cuda CUDA_ROOT=/opt/cuda CUDA_HOME=/opt/cuda
export LLVM_CONFIG_20=/workspace/scratch/llvm20/bin/llvm-config
export CARGO_TARGET_DIR=/workspace/scratch/builds/ironsand
```

## Regression check (single command, both Steps 1-3)

```sh
rm -rf /workspace/scratch/builds/ironsand/debug/build/rustc_codegen_nvvm-*
rm -rf /workspace/scratch/builds/ironsand/debug/librustc_codegen_nvvm.so
cargo build -p rustc_codegen_nvvm --features llvm20
nm --undefined-only /workspace/scratch/builds/ironsand/debug/librustc_codegen_nvvm.so \
    | grep -E ' LLVMBuildLoad$| LLVMAddGlobalDCEPass$| LLVMConstZExt$'
# Expected: no output (0 hits).
```

The matching LLVM 19 regression check:

```sh
rm -rf /workspace/scratch/builds/ironsand/debug/build/rustc_codegen_nvvm-*
rm -rf /workspace/scratch/builds/ironsand/debug/librustc_codegen_nvvm.so
cargo build -p rustc_codegen_nvvm --features llvm19
nm --undefined-only /workspace/scratch/builds/ironsand/debug/librustc_codegen_nvvm.so \
    | grep -E ' LLVMBuildLoad$| LLVMAddGlobalDCEPass$| LLVMConstZExt$'
# Expected: also no output (0 hits). The cmd-gate tightening that
# closed Step 1 also tightened LLVM 19.
```

## Open followups (separate from Steps 1-3)

### `init.rs:140` forward-pointer (Step 4)

The original spec included `init.rs:140` in the citation set, but
this is a **forward-pointer**, not a Steps 1-3 implementation
site. The 3-line breadcrumb that lives at `init.rs:146-148` is a
comment in the runtime guard message for
`LLVMContextSetDiagnosticHandler` that points *to* this recipe's
Step 4. It's the only `init.rs` content this recipe references;
the rest of `init.rs` is unrelated to the shim.

### Step 4 — `LLVMRustStringWriteImpl`

The 4th unresolved cdylib symbol under `--features llvm20` is not a
host-rustc ABI expectation — it's our missing C++ body. Declared at
`LLVMWrapper.h:27`, called at `LLVMWrapper.h:36` from
`RawRustStringOstream::write_impl`, but no body in any of our `*.cpp`
or `*.rs`. The diagnosis that established this and the LoC table for
the missing body lives in **`.dejavue/decisions.md` 2026-06-28T22:00**
[CORRECTION] entry.

### What end-to-end `--features llvm20 = runs` requires

Steps 1-3 of this recipe close the cdylib dlopen error on three of the
four unresolved symbols. Until Step 4 is also landed, end-to-end
`cargo build -p vecadd --no-default-features --features llvm20` will
continue to fail at the dlopen with `undefined symbol:
LLVMRustStringWriteImpl`. The cuda-builder feature-propagation fix in
`crates/cuda_builder/src/lib.rs` (from this same session) keeps the
inner kernels-build cargo invocation in lockstep with the active
`--features llvmXX` flag, which is a pre-condition for Steps 1-3 to
actually take effect at e2e time.

## Cross-links

- **`.dejavue/decisions.md` 2026-06-28T20:50** — "LLVM 20 plumbing:
  backend compiles clean, 4/4 examples build binaries, 0/4 can dlopen.
  Defer runtime shim as a future track" (the original plumbing
  decision; predates this recipe by an hour).
- **`.dejavue/decisions.md` 2026-06-28T22:00** — "[CORRECTION] LLVM 20
  shim: `LLVMRustStringWriteImpl` is our missing body, not a
  host-rustc ABI expectation. Scope downgrades from 3-5 days to
  ~1 day". Step 4 lives here.
- **`.dejavue/references/llvm19-llvm22-migration-recipe-archived.md`**
  (renamed post-Phase-3d from `.dejavue/references/llvm22-build-recipe.md`)
  — LLVM 22 source-compile recipe (Steps 1-6 of which must be done to land
  LLVM 22 end-to-end). Independent of this runtime-shim recipe; preserved
  for historical context now that the LLVM 22 cargo feature was dropped in
  commit 79899f8.

## Gotchas carried over

- **`cargo build` cache invalidation.** Every concrete gate edit above
  touches a `proc` macro, an `extern "C"`, or an `llvm.rs` enum, so
  after each edit, blow away the backend cache:
  `rm -rf /workspace/scratch/builds/ironsand/debug/build/rustc_codegen_
  nvvm-*`. cdylib incremental rebuilds are notorious for stale symbols.
- **`compile_error!` mutual-exclusion guard on `--features {llvm19,
  llvm20, llvm22}`** (originated in `cuda_builder/src/lib.rs`, from
  entry 10 of `.dejavue/decisions.md`) is unchanged by this recipe.
- **TODO: harden the static-assertion sentinel
  `const _: usize = b"globaldce".len();` at `src/nvvm.rs:378`.**
  Currently a no-op compile-time marker — the byte string literal is
  referenced but not compared to the explicit `9` length arg passed
  to `LLVMRustFindAndCreatePass` five lines below. Hardening target:
  swap for `static_assertions::const_assert_eq!(b"globaldce".len(), 9);`
  which requires promoting `static_assertions` from
  `rustc_codegen_nvvm`'s optional deps to its `[dev-dependencies]`.

## Success criteria

1. `cargo check -p rustc_codegen_nvvm --features llvm19` RC=0
   (verified).
2. `cargo check -p rustc_codegen_nvvm --no-default-features --features
   llvm20` RC=0 (verified).
3. `nm --undefined-only` on the cdylib after a clean build returns
   empty for `LLVMBuildLoad | LLVMAddGlobalDCEPass | LLVMConstZExt`
   under both `--features llvm19` and `--features llvm20` (verified).
4. Step 4 (`LLVMRustStringWriteImpl`) closure on the separate track
   per `.dejavue/decisions.md` 2026-06-28T22:00 [CORRECTION] entry.
5. **(Stretch)** Same-success under `--features llvm22` — once the
   LLVM 22 source-side steps (`.dejavue/references/llvm19-llvm22-migration-recipe-archived.md`
   Steps 1-6; pre-Phase-3d path was `.dejavue/references/llvm22-build-recipe.md`)
   are done, the cfg gates landed here auto-apply on the llvm22 path
   because every gate's exclusion clause covers all three non-7
   toolchains. Verified by inspection; not yet verified by build.
   **Note:** as of commit 79899f8 (Phase-3d) the llvm22 cargo feature
   was dropped, so this Stretch criterion is purely aspirational and
   depends on a future re-introduction of the llvm22 plumbing.
