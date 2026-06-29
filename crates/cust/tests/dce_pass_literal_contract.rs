//! UI tests for the compile-time typo-detection contract used in
//! `crates/rustc_codegen_nvvm/src/nvvm.rs::dce_pass`.
//!
//! The actual `dce_pass` uses `static_assertions::const_assert_eq!` to pin
//! the literal length of `GLOBAL_DCE_PASS_NAME` (`b"globaldce"`, 9 bytes)
//! at compile time. This is the const-literal branch of the audit pair
//! landed in the prior session:
//! - `f1c9f21` — centralized `b"globaldce"` into `GLOBAL_DCE_PASS_NAME`
//!   module-level const (`crates/rustc_codegen_nvvm/src/nvvm.rs`).
//! - `c15187b` — documented `AsCCharPtr`'s typed-byte-count contract
//!   on the runtime `&str` path (`crates/rustc_codegen_nvvm/src/common.rs`,
//!   `crates/rustc_codegen_nvvm/src/back.rs`).
//!
//! These tests pin the **compile-time** leg of that pair. The synthetic
//! sources under `tests/ui/dce_pass_literal/` use the SAME
//! `static_assertions::const_assert_eq!` macro as the production site so
//! the auto-blessed `.stderr` mirrors the actual diagnostic verbatim.
//! (`static_assertions = "1.1"` is added to `[dev-dependencies]` in
//! `crates/cust/Cargo.toml` for this purpose.)
//!
//! Toolchain pin: `.stderr` is coupled to `rust-toolchain.toml`. Re-bless
//! via `TRYBUILD=overwrite cargo test -p cust --test
//! dce_pass_literal_contract` after any rustup bump, before CI green.
//!
//! Coverage:
//! - Positive (`pass_*.rs`): same-length transpositions are INTENTIONALLY
//!   not caught — the documented limitation, complementing `c15187b`'s
//!   `AsCCharPtr` typed-byte-count contract.
//! - Negative (`fail_*.rs`): length-mismatch typos ARE caught at compile
//!   time and prevent the broken literal from reaching LLVM's
//!   `LLVMRustFindAndCreatePass` FFI.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/dce_pass_literal/pass_*.rs");
    t.compile_fail("tests/ui/dce_pass_literal/fail_*.rs");
}
