use libc::c_char;

/// Extension trait for explicit casts to `*const c_char`.
///
/// # Invariant
/// Every implementation MUST return a pointer valid for exactly the byte
/// count of the underlying byte sequence — NOT a null-terminated C-string.
/// Callers must pair `as_c_char_ptr()` with the byte count (e.g.
/// `pass_name.len()` for `&str`), not a C-string `strlen`. The Rust-side
/// `&str` and `&[u8]` impls both return `as_ptr().cast()` (no allocation,
/// no null-terminator added); the C++ side reads `(ptr, len)` via
/// `StringRef(ptr, len)` (see e.g.
/// `rustc_llvm_wrapper/PassWrapper.cpp::LLVMRustFindAndCreatePass`).
pub(crate) trait AsCCharPtr {
    /// Equivalent to `self.as_ptr().cast()`, but only casts to `*const c_char`.
    fn as_c_char_ptr(&self) -> *const c_char;
}

impl AsCCharPtr for str {
    // See trait-level # Invariant above.
    fn as_c_char_ptr(&self) -> *const c_char {
        self.as_ptr().cast()
    }
}

impl AsCCharPtr for [u8] {
    // See trait-level # Invariant above.
    fn as_c_char_ptr(&self) -> *const c_char {
        self.as_ptr().cast()
    }
}
