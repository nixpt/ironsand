---
type: audit
dcp: DCP/1.0
status: closed-with-open-precondition
renamed-from: llvm20-step4-allocation-audit.md
phase-3d-update: |-
  2026-06-29 — the audit target (`LLVMRustStringWriteImpl` FFI body + `RustString` struct in `crates/rustc_codegen_nvvm/src/init.rs` lines 140-185) is still current and exercising under the LLVM 19 path.
  The Phase-3d plumbing collapse did NOT touch `init.rs`.
  The unresolved ABI-alignment precondition (Section D, "audit the host-rustc-pinned rlib for the canonical `RustString` layout") remains open and is the only outstanding follow-up from this audit.
  Cross-link update: the in-body Cross-references entry that previously pointed at `.dejavue/references/llvm20-runtime-shim-recipe.md` was patched to point at `.dejavue/references/llvm19-runtime-shim-recipe.md` (the post-Phase-3d rename target).
---

# LLVM 20 Step 4 — `OpaqueRustString` Allocation-Site Audit

**Question.** After landing the Step 4 body for `LLVMRustStringWriteImpl` in
`crates/rustc_codegen_nvvm/src/init.rs` (the `#[repr(C)] RustString { string:
String }` + `unsafe extern "C" fn LLVMRustStringWriteImpl(buf, ptr, len)`
body that delegates to `push_str` over a UTF-8-cast slice), is the FFI body
an abstract cast sitting on top of nothing, or does the runtime path actually
exercise it?

## Method

`rg 'RustString|OpaqueRustString|RustStringRef'` over `crates/` (entire
workspace): **11 hits, all 11 in the new `init.rs` body**. The RustString
struct appended in Step 4 is the *only* `RustString*` identifier in the
codebase. There is *no* prior sink-allocator on the Rust side.

`rg 'LLVMRustWrite|LLVMDiagnosticInfo' -g *.rs` over `crates/`: **0 hits**.
Rust never directly invokes any of `LLVMRustWriteTypeToString` /
`LLVMRustWriteValueToString` / `LLVMRustWriteTwineToString` /
`LLVMRustUnpackOptimizationDiagnostic` / `LLVMRustWriteDiagnosticInfoToString`.

## Findings

### A. Sink allocator lives upstream of the FFI boundary

The `*mut RustString` is threaded **into** our FFI from LLVM's diagnostic
printers in
`crates/rustc_codegen_nvvm/rustc_llvm_wrapper/RustWrapper.cpp`:

- `LLVMRustWriteTypeToString(LLVMTypeRef, RustStringRef)` at `RustWrapper.cpp:1407`
- `LLVMRustWriteValueToString(LLVMValueRef, RustStringRef)` at `RustWrapper.cpp:1413`
- `LLVMRustWriteTwineToString(LLVMTwineRef, RustStringRef)` at `RustWrapper.cpp:1479`
- `LLVMRustUnpackOptimizationDiagnostic(...)` at `RustWrapper.cpp:1486` (5 RustStringRef args)
- `LLVMRustWriteDiagnosticInfoToString(...)` at `RustWrapper.cpp:1540`
- `LLVMRustWriteSMDiagnosticToString(...)` at `RustWrapper.cpp:1676`

Each builds a `RawRustStringOstream OS(Str);` and feeds it to
`Type::print(OS)` / `Value::print(OS)` / `DiagnosticInfo::print(OS)`. Whenever
LLVM's `raw_ostream::flush()` runs (function destructor), `write_impl(Ptr,
Size)` fires and calls `LLVMRustStringWriteImpl(Str, Ptr, Size)`.

**Whatever allocates `Str` is on the *caller* side of the FFI** — that's
`RawRustStringOstream`'s constructor argument, which is caller-controlled.
The canonical upstream pattern is `let mut s = String::new(); let sref: RustStringRef = &mut s as *mut _ as RustStringRef;` (any T that's `#[repr(C)]` and
sized to a single `String` field works at the pointer level; the optimization
diagnostic printer does *not* rely on `*buf`'s layout — it only stores the
pointer and feeds it back).

### B. Which Rust codegen entry points could trigger the printers

| Path | File:line | Status |
|------|-----------|--------|
| `compile_thin_module` | `lib.rs:276-283` | **`todo!()`** — never reached in this fork |
| `optimize_and_codegen_thin` | `lib.rs:311-319` | Wired through `lto::optimize_and_codegen_thin` → `_shared_emitter` received and **dropped** |
| `compile_codegen_unit` | `lib.rs:350-356` → `back::compile_codegen_unit` → `back::optimize` | **Active path.** Threads `&SharedEmitter` to LLVM's pass manager at `back.rs:163` |

The **only** path that reaches LLVM's pass-manager and could plausibly
trigger an optimization diagnostic is the `compile_codegen_unit` → `back::optimize`
chain. The 5 working examples (vecadd, gemv, gemm, matmul, async_api) all
flow through this path.

### C. Is the body exercised?

**Hot at compile time** (the symbol must resolve at dlopen) — but the
underlying `LLVMRustWrite*` → `RawRustStringOstream::write_impl` orbit is
**cold at runtime** under happy-path codegen. LLVM's diagnostic printers in
`RustWrapper.cpp` only fire when:

- A pass emits an `OptimizationRemark` / `OptimizationRemarkMissed` /
  `OptimizationRemarkAnalysis` (or `DiagnosticInfoIROptimization` ≥ LLVM 5).
- `Type::print` / `Value::print` are reached — these are diagnostic-only,
  not on the happy path of code emission.
- `DiagnosticInfo::print(DP)` runs — the `LLVMRustWriteDiagnosticInfoToString`
  C bridge prints LLVM-internal diagnostics, not user-facing ones.

For our 5 examples, none of these normally fire; the original LLVM-19  undefined
symbol was a cold-path bug papered over by rustc's ABI shadow.

### D. ABI risk (open precondition)

The Step 4 body assumes `*buf` points at memory with the layout
`#[repr(C)] struct RustString { string: String }`. If the actual upstream
allocator (whoever constructs the `RustStringRef` argument in rustc-side
diagnostic code paths) produces a different layout — e.g. a `{ data: *mut u8,
len: usize, cap: usize }` triple, or a different-sized inline-buf string —
`(*buf).as_string_mut()` will read past the allocation, producing
out-of-bounds reads + writes or UB.

**To verify ABI alignment** one would query the host-rustc version pinned by
`rust-toolchain.toml`, then read that rustc's `compiler/rustc_llvm/llvm-wrapper/RustWrapper.cpp`
to see the canonical `RustString` layout it constructs, and compare byte-for-byte
against ours. This is **work for a future session**; for now, mark the body as
defensive-but-not-e2e-confirmed.

## Net Verdict

- **The Step 4 body is not an abstract cast on top of nothing** — there is a
  real call site for it (RustWrapper.cpp:36, in `RawRustStringOstream::write_impl`).
- **It is cold-path in practice** — coded against the FFI boundary at
  dlopen-load time (so the cdylib would fail to load without it) but not
  exercised in the 5 examples' happy-path codegen.
- **One open precondition** — ABI alignment with whatever the caller side
  uses to construct `RustStringRef`. If the caller is the same host rustc
  pinned by `rust-toolchain.toml`, alignment is high-confidence because
  *we ourselves* control this crate and ours defines `#[repr(C)]` with a
  single `String` field — but the host rustc's compiler-side `RustString`
  layout is the actual contract. A 30-min probe against
  `~/.rustup/toolchains/<nightly-pinned>/lib/rustlib/<host>/lib/<...>.rlib`
  symbols would resolve this.

## Cross-references

- `.dejavue/references/llvm19-runtime-shim-recipe.md` (renamed
  post-Phase-3d from `.dejavue/references/llvm20-runtime-shim-recipe.md`)
  — Step 4 of the recipe.
- `.dejavue/decisions.md` entry "[CORRECTION] 2026-06-28T22:00:00" — supersedes
  previous 3-5 day estimate with 1-day / ~35 LoC budget for the full shim.
- `crates/rustc_codegen_nvvm/src/init.rs` lines 140-185 — the actual body.
