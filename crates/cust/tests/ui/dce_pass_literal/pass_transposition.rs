//! Positive control for the trybuild UI test in
//! `crates/cust/tests/dce_pass_literal_contract.rs`.
//!
//! This test PASSES because the length-mismatch typo-detection contract
//! (the `const_assert_eq` in `unsafe fn dce_pass` from
//! `crates/rustc_codegen_nvvm/src/nvvm.rs`) intentionally leaves
//! same-length transpositions for the next layer of defense:
//!
//! - `b"gloabldce"` is 9 bytes (same length as the correct `b"globaldce"`).
//! - The compile-time `const_assert_eq!(PASS_NAME.len(), 9)` does NOT fire.
//! - This is a known and documented limitation.
//! - Same-length typos are caught by the typed-byte-count contract on
//!   `AsCCharPtr` (see `crates/rustc_codegen_nvvm/src/common.rs`'s
//!   `# Invariant` block; audit commit `c15187b`).
//!
//! Mirrors the actual `unsafe fn dce_pass` body shape. Uses the SAME
//! `static_assertions::const_assert_eq!` macro as the production site so
//! the auto-blessed `.stderr` mirrors the actual diagnostic verbatim.

/// Module-level pass name (mirrors `GLOBAL_DCE_PASS_NAME` from
/// `crates/rustc_codegen_nvvm/src/nvvm.rs::dce_pass`).
const PASS_NAME: &[u8] = b"gloabldce";

/// Mirrors `unsafe fn dce_pass` from
/// `crates/rustc_codegen_nvvm/src/nvvm.rs` — only the const-assertion
/// shape, not the FFI call (LLVM is not available in the trybuild env).
unsafe fn dce_pass() {
    // Same-length transposition: const_assert_eq does not fire on
    // a matching length. This is the documented limitation.
    static_assertions::const_assert_eq!(PASS_NAME.len(), 9);
}

fn main() {}
