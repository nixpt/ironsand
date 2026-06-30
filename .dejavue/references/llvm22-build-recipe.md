---
type: howto
dcp: DCP/1.0
status: planned-migration  (not yet completed end-to-end)
---

# LLVM 22 Migration Recipe

Planned migration from production LLVM **19** to LLVM **22**. The cargo
plumbing is already in place (see `.dejavue/decisions.md` entry 10 — the
`--features llvm22` branch is now formally parallel to `llvm19` / `llvm20`).
This recipe documents what is **left to do** before LLVM 22 can compile
the backend, derived from the 15+ API-drift catalogue in entry 8 and the
two-line break catalogue from entry 9's first probe.

> ⚠ The `Instrumentation.h` relocation was the very first fatal under
> LLVM 22 — **already fixed** for LLVM 20 (entry 10 likewise), so it is
> not in the scope below. The remaining breaks are the migration work.

## Status (as of entry 10)

| Stage | LLVM 19 (prod) | LLVM 20 (probe) | LLVM 22 (probe) |
|---|---|---|---|
| `cargo build -p rustc_codegen_nvvm` | ✅ exit 0 | ✅ exit 0 (2 surface fixes) | ❌ first fatal at `Instrumentation.h` + 15+ to go |
| Example binaries compile | ✅ | ✅ 4/4 | ❌ backend won't link |
| Example links to cuda runtime | ✅ | ❌ unresolved C-API symbols (entry 10 dsec) | ❌ same as LLVM 20 + more |

The 2 surface fixes that already shipped under LLVM 20 plumbing
(decision 10) are the *exact* pattern to apply below: a `#if LLVM_VERSION
_MAJOR < 22` / `>= 22` cf gate that selects either the legacy form (read
from `llvm19` directory tree) or the LLVM 22 form.

## Toolchain (verified)

LLVM 22.1.8 from `release/22.x` is already extracted at
`/workspace/scratch/llvm22` (cmake + ninja, NVPTX + X86 only, -j8, ~6 GB
installed — see entry 9 for the exact cmake invocation). **Re-build the
source only if you need a fresh bitcode of `libintrinsics.bc`** (see
"libintrinsics" below; the existing `libintrinsics.bc` in tree is LLVM 19
bitcode and is incompatible with `libintrinsics_v22.bc`).

## Environment

```sh
export CUDA_PATH=/opt/cuda CUDA_ROOT=/opt/cuda CUDA_HOME=/opt/cuda
export LLVM_CONFIG_22=/workspace/scratch/llvm22/bin/llvm-config
export CARGO_TARGET_DIR=/workspace/scratch/builds/ironsand
```

## Migration steps (in order)

### Step 1 — `Attribute::NoCapture` rename  (decision entry 8)

LLVM 22's `llvm::Attribute` enum deprecates `NoCapture` in favour of an
explicit `Capture` instruction return-form. Two confirmed call sites:

| File | Line | Form |
|---|---|---|
| `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/RustWrapper.cpp` | 267 | `case NoCapture: return Attribute::NoCapture;` |
| `crates/rustc_codegen_nvvm/src/abi.rs` | 120 | `f(llvm::Attribute::NoCapture);` |

Fix pattern: confirm the LLVM 22 successor name with
`grep -n NoCapture /workspace/scratch/llvm22/include/llvm/IR/Attributes.inc`
(probe before committing — might be `Capture`, `CaptureWithin`, etc.).
Then swap `Attribute::NoCapture` for the new name on both sides; no
`#if`-gate is needed (the legacy name is only reachable from the rust-side
enum, and the rust-side enum is forward-decl-only — no consumers).

### Step 2 — `Attribute::StructRet` typed-only  (decision entry 8)

LLVM 22 drops the untyped `apply_llfn` overload on attributes that carry
a type payload (StructRet, ByVal, InAlloca). Three confirmed call sites
in `src/abi.rs`:

| File | Lines | Form |
|---|---|---|
| `src/abi.rs` | 446–449 | `llvm::Attribute::StructRet.apply_llfn_with_type(...)` ✅ already typed |
| `src/abi.rs` | 452–453 | `llvm::Attribute::StructRet.apply_llfn(...)` ❌ untyped, must go |

The wrapper at `src/llvm.rs:96` defines our own `apply_llfn` (no type
arg) and at `src/llvm.rs:101` defines `apply_llfn_with_type` — both
delegating to llvm-c. The untyped call at `src/abi.rs:452` is the only
StructRet site that needs rewriting. Drop the `else if` arm; pass `ty`
through from the caller.

Plus a scan — **MUST verify** whether `Attribute::ByVal` and
`Attribute::InAlloca` are reachable through `apply_llfn` from anywhere
else (decision 8 implies yes but the codebase grep did not surface
sites). Sweep:

```
rg -n 'Attribute::(ByVal|InAlloca).apply_llfn(/[^_])' crates/
```

For each hit, convert to `apply_llfn_with_type`. Expected count: 0-3
sites, given how narrowly the typed form is used today.

### Step 3 — `PassManagerBuilder` removal  (decision entry 8 "initialize\*")

**Confirmed by direct probe (LLVM 22 install, this session):** neither
`llvm/Transforms/IPO/PassManagerBuilder.h` (the C++ class) nor
`llvm-c/Transforms/PassManagerBuilder.h` (the C-binding header) exists
in LLVM 22:

```
$ ls /workspace/scratch/llvm22/include/llvm/Transforms/IPO/PassManagerBuilder.h
ls: cannot access ...: No such file or directory
$ ls /workspace/scratch/llvm22/include/llvm-c/Transforms/PassManagerBuilder.h
ls: cannot access ...: No such file or directory
```

So the migration is **delete the shim entirely** — there is no fallback
C API to route to. Rewrite to the new-PM API surface (`LLVMCreate
PassManager` → `LLVMAddPass` → `LLVMRunPass`).

Sites in `crates/rustc_codegen_nvvm/rustc_llvm_wrapper/PassWrapper.cpp`:
- include line 36 (`llvm/Transforms/IPO/PassManagerBuilder.h`) — gone
- include line 57 (`llvm-c/Transforms/PassManagerBuilder.h`) — gone
- opaque shim at lines 59, 77–80, 80, 96 (`LLVMOpaquePass
  ManagerBuilder`, `DEFINE_STDCXX_CONVERSION_FUNCTIONS(Pass
  ManagerBuilder, LLVMPassManagerBuilderRef)`, `unwrap(ref)`)
- entry points at lines 101 (`LLVMPassManagerBuilderCreate`), 106
  (`LLVMPassManagerBuilderDispose`), 111 (`…SetSizeLevel`), 117–118
  (`…SetDisableUnrollLoops`), 124–125 (`…UseInlinerWithThreshold`)
  — and ~5 more below the grep ceiling (~10–12 entry points total)

The shim block spans roughly lines 62–134 of the file. Each entry-point
function needs to be either **deleted entirely** (if the legacy-PM
behaviour has no functional new-PM equivalent), **rewritten using new-PM
pass enumeration** (if a thin replacement exists), or **flagged as a
deliberately-removed-from-translation** symbol.

Fix pattern:

1. **Delete** both `#include`s at lines 36 and 57.
2. **Delete** the opaque-conversion functions at lines 77–80 and 96,
   and the `struct LLVMOpaquePassManagerBuilder` at line 80.
3. **Audit every caller** of `LLVMPassManagerBuilder*` C entry points:

   | Function | New-PM replacement |
   |---|---|
   | `LLVMPassManagerBuilderCreate` | None — delete |
   | `LLVMPassManagerBuilderDispose` | None — delete (the `LLVMPassManager` it produced is also legacy) |
   | `LLVMPassManagerBuilderSetSizeLevel` | None — delete (size-level was an opt in legacy-PM only) |
   | `LLVMPassManagerBuilderSetDisableUnrollLoops` | None — set per-function inline attributes instead |
   | `LLVMPassManagerBuilderUseInlinerWithThreshold(N)` | `LLVMPassManagerAddPass(pipeline, LLVMCreateFunctionInliningPass(N))` via a new-PM-equivalent shim |
   | Any call-site in `src/back.rs` referencing the legacy pipeline mode | Rewrite as new-PM-only |

4. **Audit `crates/rustc_codegen_nvvm/src/back.rs`** for callers of
   `LLVMRustThinLTOPassManagerRun` style functions — these references
   are the "deleted legacy-PM initialize\* calls" that entry 8
   catalogued. Each caller either gets deleted, replaced with
   `LLVMCreateThinLTOData`, or routed to the new-PM backend.

This Step alone is larger than my initial estimate suggested. It is the
single biggest mechanical-repair block in the recipe; allocate
**70–90 LoC** here.

### Step 4 — Triple typed-API verification  (decision entry 8)

Entry 8's blanket "*Triple-typed APIs*" label was previously read as a
`getArch() → getArchType()` rename — but **direct probe of LLVM 22
confirms `Triple::getArch()` is still canonical** at
`/workspace/scratch/llvm22/include/llvm/TargetParser/Triple.h:417`,
alongside `getArchName()` (string form) and the various `isX86()` /
`isNVPTX()` predicates. The Two callsites in our wrapper at
`PassWrapper.cpp:460-461` are **not** breaking.

What entry 8 most likely meant: predicate methods were renamed or their
return-type tightened (e.g., previously `bool isArch(ArchType)`,
now `bool isArch()` with an implicit `this`). Audit, do not rewrite
blindly:

| File | Line | Form | Status |
|---|---|---|---|
| `rustc_llvm_wrapper/RustWrapper.cpp` | 200 | `Triple::normalize(Triple)` | ✅ Same in LLVM 22 |
| `rustc_llvm_wrapper/RustWrapper.cpp` | 1844 | `Triple TargetTriple(unwrap(M)->getTargetTriple());` | ✅ Same |
| `rustc_llvm_wrapper/PassWrapper.cpp` | 143 | `Triple TargetTriple(Builder->TargetTriple);` | ✅ Same |
| `rustc_llvm_wrapper/PassWrapper.cpp` | 460 | `Triple::ArchType HostArch = Triple(sys::getProcessTriple()).getArch();` | ✅ `getArch()` valid in LLVM 22 |
| `rustc_llvm_wrapper/PassWrapper.cpp` | 461 | `Triple::ArchType TargetArch = Target->getTargetTriple().getArch();` | ✅ `getArch()` valid in LLVM 22 |

So this Step is **investigation-led, not necessarily a rewrite**. After
this audit passes, the LoC for Step 4 may legitimately be 0. If a
predicate call did get renamed (e.g., `TargetTriple.isArch
(Triple::x86)` → `TargetTriple.getArch() == Triple::x86`), the fix is
to convert each renamed callsite individually.

Note: `Triple.h` was already migrated to `TargetParser/Triple.h` in
earlier (LLVM 19) plumbing at `rustllvm.h:20-22` — *not* as part of
LLVM 20 plumbing. The attribution "entry 10 already applied the include
fix" is **wrong**; the include migration predates entry 10.

### Step 5 — `PassManagerBuilder` zero-warning compile

This is the cleanup for `PassWrapper.cpp` that follows from Step 3.
After Step 3 ships there are no deprecation warnings — verify by:

```sh
cargo build -p rustc_codegen_nvvm --features llvm22 2>&1 \
  | grep -E 'warning.*PassManagerBuilder|warning.*legacy'
```

Expected: zero matches. If warnings remain, the C++ class is leaking
through some second-order include (e.g., `llvm/Transforms/IPO.h`)
that pulls in `PassManagerBuilder` transitively — strip those headers
and the warnings vanish.

### Step 6 — `libintrinsics.bc` regeneration

The `libintrinsics.bc` in tree is LLVM 19 bitcode. The build script's
`--features llvm22` branch assembles against the LLVM 22 `llvm-as` and
**needs the .ll source re-compiled**. Steps:

1. Source-of-truth is `libintrinsics.ll` (LLVM 19 formatted).
2. `llvm22_as` (from `/workspace/scratch/llvm22/bin/llvm-as`) is fed
   the .ll → emits `libintrinsics_v22.bc`.
3. Update `crates/rustc_codegen_nvvm/build.rs`'s `configure_lib
   intrinsics(Box::new(LlvmConfig22 { … }))` (the existing `llvm20`
   variant at entry 10 is the canonical template) to:
   - emit `libintrinsics_v22.bc` via `llvm22_as`,
   - register the artifact in `Linker::add_library("libintrinsics")`
     linker pipeline (no API change needed here — Linker takes the bc
     blob by stem).

### Step 7 — End-to-end verification

After Steps 1–6:

```sh
# Backend
cargo build -p rustc_codegen_nvvm --features llvm22          # → 0

# Examples build binaries (--no-default-features so only llvm22 active)
cargo build -p vecadd -p gemm -p gemv -p attn --no-default-features --features llvm22

# Run + correctness check
LD_LIBRARY_PATH=/opt/cuda/lib64 /workspace/scratch/builds/ironsand/debug/vecadd
```

The `cuda_builder::compile_error!` guard (entry 10) will fire with a
clear error message if both `llvm19` and `llvm22` are active — that is
intentional, treat it as PASS-side, not REGRESSION.

## LoC estimate (mirrors entry 8)

Per entry 8: "80–150 LoC of mechanical repair across `PassWrapper.cpp`
+ `RustWrapper.cpp`." Realistic split after LLVM 20's work removed 2:

| Step | Estimate |
|---|---|
| 1 NoCapture rename | ~3 LoC |
| 2 StructRet typed-only | ~5 LoC |
| 3 `PassManagerBuilder` removal | **~70–90 LoC** (largest block — shim spans PassWrapper.cpp lines 62–134) |
| 4 Triple typed-API audit | **~0–3 LoC** (likely no rewrite) |
| 5 Zero-warning compile | 0 LoC (overhead from Step 3) |
| 6 libintrinsics.bc regen | ~10 LoC (build.rs only) |
| Buffer for unexpected API drift | ~30 LoC |
| **Total** | **~120–150 LoC** (now inside entry 8's "80–150" upper range) |

## Post-migration: opaque-pointer shim work  (decision entry 10 dsec)

`decision entry 10` documents 4 unresolved symbols at dlopen time
(under LLVM 20): `LLVMBuildLoad`, `LLVMConstZExt`, `LLVMAddGlobalDCE
Pass`, `LLVMRustStringWriteImpl`. If those persist under LLVM 22 after
this recipe is applied — likely; they are host-rustc-vendored rather
than toolchain-version-correlated — they need a separate opaque-pointer
shim plan:

1. opaque-pointer `LLVMBuildLoad2` shim — requires caller-side IR
   type recovery (currently impossible from the C interface alone; use
   the ElementType-as-cast pattern in LLVM 17+ codegen stacks).
2. `LLVMConstZExt` rename/move forwarder.
3. `LLVMAddGlobalDCEPass` deletion — the new-PM equivalent is
   `FunctionAnalytic: GlobalDCE` run as a module pass; rewrite the
   pipeline call site.
4. `LLVMRustStringWriteImpl` — audit `crates/rustc_codegen_nvvm/rustc_
   llvm_wrapper/RustWrapper.cpp` for the missing definition; if absent,
   add it back, otherwise verify the missing reference.

Estimated **3–5 extra days** focused engineering on top of this recipe
to claim end-to-end LLVM 22 success. Plan for both work blocks before
declaring the migration done.

## Gotchas

- **Cargo feature gate**: even after migration works, the user invokes
  the LLVM 22 backend with `cargo build -p vecadd --no-default-features
  --features llvm22`. The `cuda_builder::compile_error!` guard (entry
  10) requires `default = ["llvm19"]` (set at each example's
  `Cargo.toml`) to be **disabled** — i.e. always use
  `--no-default-features --features llvm22` together, never just
  `--features llvm22` (that activates both features).
- **Both `llvm19` and `llvm22`** = both-features-active = the
  `compile_error!` will burn with the expected error message — that
  is intentional, not a bug.
- **Sample Cargo.toml entries** (mirror in any new example): `default
  = ["llvm19"]` plus `llvm22 = ["cuda_builder/llvm22"]`. Examples
  `vecadd`/`gemm`/`gemv`/`attn` already have this shape for `llvm20`
  — replicate for `llvm22`.
- **Cross-check decisions.md entry 10** before claiming Steps 1-6 are
  fully completed: entry 10 says LLVM 20 plumbing mirrors LLVM 22
  architecture, so Steps already shipped under `--features llvm20`
  (the 2 surface fixes) re-apply cleanly to `--features llvm22`. The
  remaining ~120-150 LoC (see table above) is **truly new** for the
  LLVM 22 scope — Steps 1, 2, 3, 4, 6 are net-new work.
- **Don't expect a single integrated build**: each Step has its own
 1-3-file LoC footprint and its own verification probe. Land them
  incrementally, rebooting the cargo build with `rm -rf
  /workspace/scratch/builds/ironsand/debug/build/rustc_codegen_nvvm-*`
  after each Step's C++ edit so the change really rebuilds.
