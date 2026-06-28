//! UI tests for the `#[derive(KernelDescriptor)]` proc macro.
//!
//! These tests use `trybuild` to verify that:
//! - Valid structs with `#[kernel_name = "..."]` compile successfully.
//! - Missing `#[kernel_name]` produces a clear compile error.
//! - Generic structs are rejected.
//! - Enums and unions are rejected.

#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/pass_*.rs");
    t.compile_fail("tests/ui/fail_*.rs");
}
