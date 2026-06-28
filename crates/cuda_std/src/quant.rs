//! Quantization primitives for GGUF k-quant and BitNet formats.
//!
//! Provides block layout types, dequantization helpers, and integer-dot
//! primitives used by Q4_K, Q6_K, and ternary (i2_s) weight formats.
//!
//! # Block layouts
//!
//! | Format | Block size | Weights/block | Bits/weight |
//! |--------|-----------|---------------|-------------|
//! | Q4_K   | 144 bytes | 256           | 4.5         |
//! | Q6_K   | 210 bytes | 256           | 6.56        |
//! | Ternary| 4 bytes   | 16 (packed)   | 2.0         |
//!
//! All layouts are byte-faithful to llama.cpp's GGUF packers.
//!
//! # Example
//!
//! ```ignore
//! use cuda_std::quant::BlockQ4K;
//!
//! // `aptr` points to a contiguous buffer of Q4_K super-blocks.
//! let block = unsafe { &*(aptr.add(bbase) as *const BlockQ4K) };
//! let d = unsafe { cvt_f16((block.d[0] as u16) | ((block.d[1] as u16) << 8)) };
//! ```

use crate::warp;

// =============================================================================
// Block type definitions
// =============================================================================

/// Q4_K super-block layout (144 bytes).
///
/// Faithful to llama.cpp `block_q4_K`. Each super-block covers 256 weights.
///
/// ```text
///   d     : f16            super-block scale (scale of the sub-scales)
///   dmin  : f16            super-block min   (scale of the sub-mins)
///   scales: [u8; 12]       8 × 6-bit sub-scale + 8 × 6-bit sub-min, bit-packed
///   qs    : [u8; 128]      256 × 4-bit quants (low/high nibbles per 64 group)
/// ```
///
/// Each 32-weight sub-block has a 6-bit scale `sc` and 6-bit min `mn`;
/// dequantization is affine: `w = d·sc·q − dmin·mn`, where `q ∈ [0,15]`.
#[repr(C)]
pub struct BlockQ4K {
    /// Super-block scale (scale of the sub-scales), stored as little-endian f16.
    pub d: [u8; 2],
    /// Super-block min (scale of the sub-mins), stored as little-endian f16.
    pub dmin: [u8; 2],
    /// 8 × 6-bit sub-scale + 8 × 6-bit sub-min, bit-packed.
    pub scales: [u8; 12],
    /// 256 × 4-bit quants. Low/high nibbles per 64-weight group.
    pub qs: [u8; 128],
}

impl BlockQ4K {
    /// Bytes per super-block.
    pub const SIZE: usize = 144;
    /// Weights quantized per super-block.
    pub const K: usize = 256;
}

/// Q6_K super-block layout (210 bytes).
///
/// Faithful to llama.cpp `block_q6_K`. Each super-block covers 256 weights.
///
/// ```text
///   ql    : [u8; 128]   low 4 bits of each 6-bit quant
///   qh    : [u8;  64]   high 2 bits (4 per byte)
///   scales: [i8;  16]   16 × int8 sub-block scales (one per 16 weights)
///   d     : f16         super-block scale
/// ```
///
/// A weight's 6-bit value `q ∈ [0,63]` is `(ql_nibble) | (qh_2bits << 4)`;
/// dequantization is `w = d · scale[sb] · (q − 32)`.
#[repr(C)]
pub struct BlockQ6K {
    /// Low 4 bits of each 6-bit quant (128 bytes).
    pub ql: [u8; 128],
    /// High 2 bits (4 per byte, 64 bytes).
    pub qh: [u8; 64],
    /// 16 × int8 sub-block scales (one per 16 weights).
    pub scales: [i8; 16],
    /// Super-block scale, stored as little-endian f16.
    pub d: [u8; 2],
}

impl BlockQ6K {
    /// Bytes per super-block.
    pub const SIZE: usize = 210;
    /// Weights quantized per super-block.
    pub const K: usize = 256;
}

// =============================================================================
// Low-level inline-PTX primitives
// =============================================================================

/// `dp4a.s32.s32`: `c + Σ s8x4(a)·s8x4(b)` as `i32`. Requires sm_61+.
///
/// This is the GPU's 4-way int8 dot-product accumulation instruction,
/// used heavily in W8A8 and quantized-GEMV kernels.
#[cfg(target_os = "cuda")]
#[inline(always)]
pub unsafe fn dp4a(a: u32, b: u32, c: i32) -> i32 {
    let d: i32;
    unsafe {
        core::arch::asm!(
            "dp4a.s32.s32 {d}, {a}, {b}, {c};",
            d = out(reg32) d,
            a = in(reg32) a,
            b = in(reg32) b,
            c = in(reg32) c,
        )
    };
    d
}

#[cfg(not(target_os = "cuda"))]
#[inline(always)]
pub unsafe fn dp4a(_a: u32, _b: u32, _c: i32) -> i32 {
    0
}

/// Convert f16 bits to f32 via PTX `cvt.f32.f16`.
#[cfg(target_os = "cuda")]
#[inline(always)]
pub unsafe fn cvt_f16(bits: u16) -> f32 {
    let o: f32;
    unsafe {
        core::arch::asm!(
            "cvt.f32.f16 {o}, {i};",
            o = out(reg32) o,
            i = in(reg16) bits,
        )
    };
    o
}

#[cfg(not(target_os = "cuda"))]
#[inline(always)]
pub unsafe fn cvt_f16(_bits: u16) -> f32 {
    0.0
}

/// Convert f32 to f16 bits (round-to-nearest) via PTX `cvt.rn.f16.f32`.
#[cfg(target_os = "cuda")]
#[inline(always)]
pub unsafe fn f16_bits(v: f32) -> u16 {
    let o: u16;
    unsafe {
        core::arch::asm!(
            "cvt.rn.f16.f32 {o}, {i};",
            o = out(reg16) o,
            i = in(reg32) v,
        )
    };
    o
}

#[cfg(not(target_os = "cuda"))]
#[inline(always)]
pub unsafe fn f16_bits(_v: f32) -> u16 {
    0
}

/// Little-endian `u16` load from a byte pointer at `offset`.
#[inline(always)]
pub unsafe fn load_u16(p: *const u8, off: usize) -> u16 {
    let lo = unsafe { *p.add(off) } as u16;
    let hi = unsafe { *p.add(off + 1) } as u16;
    lo | (hi << 8)
}

// =============================================================================
// Q4_K helpers
// =============================================================================

/// Unpack sub-block `j`'s 6-bit scale and min from the 12-byte `scales` array.
///
/// `sc` must point to the first byte of a `BlockQ4K::scales` array.
/// This is the llama.cpp `get_scale_min_k4` logic.
#[inline(always)]
pub unsafe fn scale_min(j: usize, sc: *const u8) -> (u32, u32) {
    unsafe {
        if j < 4 {
            ((*sc.add(j) & 63) as u32, (*sc.add(j + 4) & 63) as u32)
        } else {
            let d = ((*sc.add(j + 4) & 0xF) | ((*sc.add(j - 4) >> 6) << 4)) as u32;
            let m = ((*sc.add(j + 4) >> 4) | ((*sc.add(j) >> 6) << 4)) as u32;
            (d, m)
        }
    }
}

/// Unpack the 12-byte `scales` array from 3 `u32`s into two `[u32; 8]` arrays.
///
/// The 12 bytes hold 8 × 6-bit sub-scale values and 8 × 6-bit sub-min values,
/// bit-packed per llama.cpp's `block_q4_K` packer. This hoists all 12 bytes
/// into registers up front so per-sub-block dot loops have no scale-unpack
/// latency.
///
/// `s0` = bytes [0..4), `s1` = bytes [4..8), `s2` = bytes [8..12). Little-endian.
#[inline(always)]
pub fn unpack_q4k_scales(s0: u32, s1: u32, s2: u32) -> ([u32; 8], [u32; 8]) {
    let sc = [
        s0 & 0x3F,
        (s0 >> 8) & 0x3F,
        (s0 >> 16) & 0x3F,
        (s0 >> 24) & 0x3F,
        (s2 & 0xF) | (((s0 >> 6) & 0x3) << 4),
        ((s2 >> 8) & 0xF) | (((s0 >> 14) & 0x3) << 4),
        ((s2 >> 16) & 0xF) | (((s0 >> 22) & 0x3) << 4),
        ((s2 >> 24) & 0xF) | (((s0 >> 30) & 0x3) << 4),
    ];
    let mn = [
        s1 & 0x3F,
        (s1 >> 8) & 0x3F,
        (s1 >> 16) & 0x3F,
        (s1 >> 24) & 0x3F,
        ((s2 >> 4) & 0xF) | (((s1 >> 6) & 0x3) << 4),
        ((s2 >> 12) & 0xF) | (((s1 >> 14) & 0x3) << 4),
        ((s2 >> 20) & 0xF) | (((s1 >> 22) & 0x3) << 4),
        ((s2 >> 28) & 0xF) | (((s1 >> 30) & 0x3) << 4),
    ];
    (sc, mn)
}

// =============================================================================
// Ternary / BitNet helpers
// =============================================================================

/// Spread 4 packed 2-bit codes (one byte: `c0|c1<<2|c2<<4|c3<<6`) into 4 byte
/// lanes (`c0 | c1<<8 | c2<<16 | c3<<24`), each value still in `{0,1,2}`.
///
/// Pure mask/shift/or — no per-byte subtract (that would borrow across bytes),
/// so the `−1` of `signed = code − 1` is deferred to a single `Σx` correction.
#[inline(always)]
pub fn spread(b: u32) -> u32 {
    (b & 0x03) | ((b & 0x0C) << 6) | ((b & 0x30) << 12) | ((b & 0xC0) << 18)
}

// =============================================================================
// Warp reduction helpers (commonly used alongside quantization)
// =============================================================================

/// Butterfly all-reduce of an `f32` across the warp.
///
/// Uses [`warp::warp_shuffle_xor`] with halving offsets. Equivalent to
/// `warp_reduce(mask, v, Add)` but with a smaller API surface.
#[inline(always)]
pub unsafe fn warp_sum_f32(mut v: f32) -> f32 {
    const WARP: u32 = 32;
    let mut off = WARP / 2;
    while off >= 1 {
        let (bits, _) = unsafe { warp::warp_shuffle_xor(u32::MAX, v.to_bits(), off, WARP) };
        v += f32::from_bits(bits);
        off >>= 1;
    }
    v
}

/// Butterfly all-reduce of an `i32` across the warp.
#[inline(always)]
pub unsafe fn warp_sum_i32(mut v: i32) -> i32 {
    const WARP: u32 = 32;
    let mut off = WARP / 2;
    while off >= 1 {
        let (bits, _) = unsafe { warp::warp_shuffle_xor(u32::MAX, v as u32, off, WARP) };
        v += bits as i32;
        off >>= 1;
    }
    v
}
