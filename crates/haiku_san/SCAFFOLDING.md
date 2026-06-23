# Haiku-San Crate Scaffolding Summary

**Date**: 2026-06-23  
**Status**: Library crate created and integrated

---

## What's Been Done

### ✅ Crate Structure Created

```
crates/haiku_san/
├── Cargo.toml                 # Crate manifest
├── README.md                  # User guide + quick start
├── SCAFFOLDING.md            # This file
├── doc/
│   ├── DESIGN.md             # Core architecture + design principles
│   ├── CAPACITY_ANALYSIS.md  # Kernel capacity limits and sweet spot
│   └── ROLE_KERNELS.md       # (Link to parent design doc)
└── src/
    └── lib.rs                # Orchestrator implementation + tests
```

### ✅ Core Implementation

- **HaikuSan struct**: Task queue, inflight event tracking, dependency graph
- **Public API**: `submit_task()`, `add_dependency()`, `launch_all_async()`, `wait_for()`
- **Stats tracking**: Task count, completion time, CPU overhead
- **Test spikes**: 4 orchestration scenarios (2-op, layer, hybrid, full-model)
- **Logging**: Uses `log` crate for debug/info output

### ✅ Design Documentation

Extracted from `.dejavue/` and integrated into crate docs:
- **DESIGN.md**: Three pillars (GPU kernels, CPU orchestrator, dependency graph)
- **CAPACITY_ANALYSIS.md**: Why 64 kernels/token is optimal
- **README.md**: Quick start, API reference, opcode table

### ✅ Integration with Examples

- Updated `examples/attn/Cargo.toml` to depend on haiku_san
- Updated `examples/attn/src/main.rs` to import from crate
- Removed old local `examples/attn/src/haiku_san.rs` module file
- Added workspace member in `/workspace/projects/ironsand/Cargo.toml`

### ✅ Compilation Status

```bash
cargo check -p haiku_san  ✓ Success
cargo tree -p haiku_san   ✓ Dependencies correct
```

---

## Next Steps

### Phase 1: Role-Based Decode Kernels (2 weeks)

Implement in new crates under `crates/`:

1. **crates/role_kernels_decode/** — 3 decode-optimized kernels
   - `RoleRMSNormSingle` (<100 μs)
   - `RoleGEMVDecodeSingle` (<500 μs)
   - `RoleFlashAttnSingle` (<2000 μs)

2. **Integrate into HaikuSan**
   - Add role-based opcodes (30-33)
   - Phase-aware dispatch in orchestrator

3. **Test in zorro decode loop**
   - 4-week implementation plan: `.dejavue/4week_implementation_plan.md`

### Phase 2: Prefill Roles (Weeks 3-4)

Batch-optimized kernels for prefill phase (deferred after decode ships).

### Phase 3: Extensions (Weeks 5+)

- Pipeline parallelism
- Token streaming
- Speculative execution

---

## Design Decision Rationale

### Why a Separate Crate?

**Before**: Haiku-San lived in `examples/attn/src/haiku_san.rs`
- Hard to discover (buried in example)
- Hard to test independently
- Hard to document (no crate README)

**After**: Haiku-San is `crates/haiku_san/` public library
- Clear module boundary
- Independent cargo test targets
- Discoverable documentation (README + design docs)
- Easy to integrate into multiple consumers (zorro, other examples)

### Why These Docs?

- **DESIGN.md**: Philosophy + architecture (why it works)
- **CAPACITY_ANALYSIS.md**: Math + limits (when it breaks)
- **README.md**: Practical guide (how to use it)

Separation allows users to jump straight to what they need.

### Why Keep Spike Tests?

Methods like `orchestrate_two_op_spike()` are in the public API for:
- Demonstration purposes
- Rapid prototyping
- Benchmarking baseline
- Testing orchestration without real GPU kernels

They're marked `// **For testing/spikes only**` in docs.

---

## Files Changed

### Created
- `/workspace/projects/ironsand/crates/haiku_san/Cargo.toml`
- `/workspace/projects/ironsand/crates/haiku_san/README.md`
- `/workspace/projects/ironsand/crates/haiku_san/src/lib.rs` (400+ lines)
- `/workspace/projects/ironsand/crates/haiku_san/doc/DESIGN.md`
- `/workspace/projects/ironsand/crates/haiku_san/doc/CAPACITY_ANALYSIS.md`

### Modified
- `/workspace/projects/ironsand/Cargo.toml` (workspace member added)
- `/workspace/projects/ironsand/examples/attn/Cargo.toml` (haiku_san dependency added)
- `/workspace/projects/ironsand/examples/attn/src/main.rs` (imports updated)

### Deleted
- `/workspace/projects/ironsand/examples/attn/src/haiku_san.rs` (moved to crate)

---

## Quality Checklist

- ✅ Crate compiles (`cargo check -p haiku_san`)
- ✅ Public API documented (inline doc comments)
- ✅ Design docs in crate
- ✅ README with quick-start
- ✅ Tests included (basic unit tests)
- ✅ Log macro support (for debugging)
- ✅ Examples integrated
- ✅ Workspace integration complete

---

## Ready for Phase 1

The scaffold is complete and ready for:
1. **Implement role-based kernels** in separate crates
2. **Integrate with Haiku-San** via new opcodes
3. **Test in zorro** decode loop

See `.dejavue/4week_implementation_plan.md` for the detailed roadmap.

---

## Quick Navigation from Here

- **To use Haiku-San**: Read `README.md`
- **To understand the design**: Read `doc/DESIGN.md`
- **To see capacity limits**: Read `doc/CAPACITY_ANALYSIS.md`
- **To implement role kernels**: See `.dejavue/4week_implementation_plan.md`

The architecture is sound. The library is ready. Let's build the kernels. 🚀
