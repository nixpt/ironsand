//! Negative control for the trybuild UI test in
//! `crates/cust/tests/dce_pass_literal_contract.rs`.
//!
//! This test FAILS TO COMPILE because the length-mismatch typo-detection
//! contract (the `const_assert_eq` in `unsafe fn dce_pass` from
//! `crates/rustc_codegen_nvvm/src/nvvm.rs`) catches typos that change the
//! literal length:
//!
//! - `b"gobaldce"` is 8 bytes (the second `l` of `globaldce` is dropped).
//! - The compile-time `const_assert_eq!(PASS_NAME.len(), 9)` MUST fire
//!   at compile time, before any LLVM FFI call is issued.
//! - Without this assertion, a typo of this form would silently reach
//!   `LLVMRustFindAndCreatePass` and return a NULL pass handle at runtime.
//!
//! Mirrors the actual `unsafe fn dce_pass` body shape. Uses the SAME
//! `static_assertions::const_assert_eq!` macro as the production site so
//! the auto-blessed `.stderr` mirrors the actual diagnostic verbatim.

/// Module-level pass name (mirrors `GLOBAL_DCE_PASS_NAME` from
/// `crates/rustc_codegen_nvvm/src/nvvm.rs::dce_pass`).
const PASS_NAME: &[u8] = b"gobaldce";

/// Mirrors `unsafe fn dce_pass` from
/// `crates/rustc_codegen_nvvm/src/nvvm.rs` — only the const-assertion
/// shape, not the FFI call (LLVM is not available in the trybuild env).
unsafe fn dce_pass() {
    // Length-mismatch typo: const_assert_eq fires at compile time on
    // the length assertion `8 == 9`. This is exactly the protection the
    // actual production `const_assert_eq` provides.
    static_assertions::const_assert_eq!(PASS_NAME.len(), 9);
}

fn main() {}
