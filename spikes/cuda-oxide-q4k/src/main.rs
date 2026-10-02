//! IRONSAND-OXIDE-1 spike — Q4_K decode GEMV on cuda-oxide vs ironsand.
//!
//! Single-source (host + device in one file, one `cargo oxide build`).
//!
//! The device kernel `gemv_q4k_fast` is a faithful port of ironsand's
//! `examples/gemv/kernels/src/gemv_q4k.rs::gemv_q4k_fast`: warp-per-row, the
//! super-block `d`/`dmin` decoded once per 256 weights, an inner loop over the
//! 8 sub-blocks with coalesced nibble reads, and a butterfly warp reduction.
//! Launch shape matches ironsand: block = 256 (8 warps), grid = m.div_ceil(8).
//!
//! The harness, on the same deterministic Q4_K bytes + x as zorro's
//! `cuda_q4_k_gemv_ironsand_matches_cpu` test, runs three paths:
//!   (a) CPU reference dequant·dot (byte-faithful to llama.cpp block_q4_K),
//!   (b) this cuda-oxide kernel,
//!   (c) the EXISTING ironsand PTX (`gemv_q4k_fast`) loaded from a `.ptx` file
//!       via cuda-core's module-from-PTX path (set IRONSAND_PTX).
//! It reports max-abs + rel-L2 for b-vs-a and b-vs-c, then a matched A/B perf
//! table (CUDA events, warmup + >=50 runs, median) for b vs c.

use cuda_core::simt::LaunchConfig;
use cuda_core::sys::{CUdeviceptr, CUevent_flags_enum_CU_EVENT_DEFAULT};
use cuda_core::{CudaContext, CudaStream, DeviceBuffer};
use cuda_device::{DisjointSlice, cuda_module, kernel, thread, warp};
use std::error::Error;
use std::ffi::c_void;

// =============================================================================
// DEVICE: Q4_K GEMV (faithful ironsand `gemv_q4k_fast` port)
// =============================================================================

#[cuda_module]
mod kernels {
    use super::*;
    use cuda_device::convert::cvt_f32_f16x2_lo;

    /// Little-endian `u16` from a byte pointer at `off`.
    #[inline(always)]
    unsafe fn load_u16(p: *const u8, off: usize) -> u16 {
        let lo = unsafe { *p.add(off) } as u16;
        let hi = unsafe { *p.add(off + 1) } as u16;
        lo | (hi << 8)
    }

    /// llama.cpp `get_scale_min_k4`: sub-block `j`'s 6-bit scale + min.
    #[inline(always)]
    unsafe fn scale_min(j: usize, sc: *const u8) -> (u32, u32) {
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

    /// Warp-per-row Q4_K GEMV. `a` = m*(k/256) 144-byte super-blocks,
    /// `x` = length k, `y` = length m. Decode `d`/`dmin` once per super-block.
    #[kernel]
    pub fn gemv_q4k_fast(
        a: &[u8],
        x: &[f32],
        mut y: DisjointSlice<f32>,
        m: usize,
        k: usize,
        beta: f32,
    ) {
        const WARP: usize = 32;
        const BLK: usize = 144;

        let gid = thread::index_1d().get();
        let row = gid / WARP;
        let lane = gid % WARP;
        if row >= m {
            return; // whole warp returns together (all 32 lanes share `row`)
        }
        let nb = k / 256;
        let row_base = row * nb * BLK;
        let aptr = a.as_ptr();
        let xptr = x.as_ptr();

        let mut acc = 0.0f32;
        let mut b = 0usize;
        while b < nb {
            let bbase = row_base + b * BLK;
            let d = cvt_f32_f16x2_lo(unsafe { load_u16(aptr, bbase) } as u32);
            let dmin = cvt_f32_f16x2_lo(unsafe { load_u16(aptr, bbase + 2) } as u32);
            let scbase = bbase + 4;

            let mut sub = 0usize;
            while sub < 8 {
                let (sc, mn) = unsafe { scale_min(sub, aptr.add(scbase)) };
                let d_eff = d * sc as f32;
                let m_eff = dmin * mn as f32;
                let g = sub >> 1;
                let qbase = bbase + 16 + g * 32;
                let byte = unsafe { *aptr.add(qbase + lane) }; // coalesced
                let nib = if (sub & 1) == 1 { byte >> 4 } else { byte & 0xF };
                let gw = b * 256 + sub * 32 + lane;
                acc += (d_eff * (nib as f32) - m_eff) * unsafe { *xptr.add(gw) };
                sub += 1;
            }
            b += 1;
        }

        let sum = warp::reduce_sum_f32(acc);
        if lane == 0 {
            // SAFETY: only lane 0 writes, and each warp owns a distinct
            // `row < m`, so writes are unique across the launch.
            let e = unsafe { y.get_unchecked_mut(row) };
            *e = sum + beta * *e;
        }
    }
}

// =============================================================================
// HOST: deterministic data, CPU oracle, launchers, correctness + perf
// =============================================================================

const BLK: usize = 144; // bytes per Q4_K super-block

/// Deterministic Q4_K weight bytes, byte-identical to zorro's
/// `cuda_q4_k_gemv_ironsand_matches_cpu` generator.
fn build_weights(rows: usize, cols: usize) -> Vec<u8> {
    let bpr = cols / 256;
    let nbytes = rows * bpr * BLK;
    let mut w: Vec<u8> = (0..nbytes).map(|i| ((i * 131 + 17) % 251) as u8).collect();
    for sb in 0..rows * bpr {
        let o = sb * BLK;
        let d = half::f16::from_f32(0.015 + 0.003 * (sb % 5) as f32);
        let dmin = half::f16::from_f32(0.008 + 0.002 * (sb % 3) as f32);
        w[o..o + 2].copy_from_slice(&d.to_le_bytes());
        w[o + 2..o + 4].copy_from_slice(&dmin.to_le_bytes());
    }
    w
}

fn build_x(cols: usize) -> Vec<f32> {
    (0..cols).map(|i| ((i % 23) as f32 - 11.0) * 0.05).collect()
}

/// llama.cpp `get_scale_min_k4` on host (matches zorro `cpu_gsm_k4`).
fn gsm_k4(j: usize, s: &[u8]) -> (u8, u8) {
    if j < 4 {
        (s[j] & 63, s[j + 4] & 63)
    } else {
        (
            (s[j + 4] & 0x0F) | ((s[j - 4] >> 6) << 4),
            (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
        )
    }
}

/// CPU reference dequant, byte-faithful to zorro's `cpu_dequant_q4k`.
fn cpu_dequant_q4k(src: &[u8], n: usize) -> Vec<f32> {
    let mut dst = vec![0f32; n];
    let nb = src.len() / BLK;
    for sb in 0..nb {
        let o = sb * BLK;
        let d = half::f16::from_le_bytes([src[o], src[o + 1]]).to_f32();
        let dmin = half::f16::from_le_bytes([src[o + 2], src[o + 3]]).to_f32();
        let scales = &src[o + 4..o + 16];
        let qs = &src[o + 16..o + BLK];
        let base = sb * 256;
        let (mut q_idx, mut out_idx, mut is) = (0usize, 0usize, 0usize);
        for _ in 0..4 {
            let (sc1, m1) = gsm_k4(is, scales);
            let (sc2, m2) = gsm_k4(is + 1, scales);
            let (d1, min1) = (d * sc1 as f32, dmin * m1 as f32);
            let (d2, min2) = (d * sc2 as f32, dmin * m2 as f32);
            for l in 0..32 {
                if base + out_idx + l < n {
                    dst[base + out_idx + l] = d1 * (qs[q_idx + l] & 0x0F) as f32 - min1;
                }
                if base + out_idx + 32 + l < n {
                    dst[base + out_idx + 32 + l] = d2 * (qs[q_idx + l] >> 4) as f32 - min2;
                }
            }
            q_idx += 32;
            out_idx += 64;
            is += 2;
        }
    }
    dst
}

fn cpu_gemv(w: &[f32], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    (0..rows)
        .map(|r| (0..cols).map(|c| w[r * cols + c] * x[c]).sum())
        .collect()
}

/// (max-abs, rel-L2) between `got` and `ref_`.
fn err(got: &[f32], ref_: &[f32]) -> (f32, f32) {
    let max_abs = got
        .iter()
        .zip(ref_)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    let num: f64 = got
        .iter()
        .zip(ref_)
        .map(|(a, b)| ((a - b) as f64).powi(2))
        .sum();
    let den: f64 = ref_.iter().map(|b| (*b as f64).powi(2)).sum();
    let rel_l2 = if den > 0.0 {
        (num / den).sqrt() as f32
    } else {
        num.sqrt() as f32
    };
    (max_abs, rel_l2)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

/// Raw launch of an ironsand-ABI `gemv_q4k_fast` entry: slices are (ptr,len)
/// pairs, `y` is a bare pointer, then m,k,beta.
#[allow(clippy::too_many_arguments)]
unsafe fn launch_ironsand(
    func: &cuda_core::simt::CudaFunction,
    stream: &CudaStream,
    a: &DeviceBuffer<u8>,
    x: &DeviceBuffer<f32>,
    y: &DeviceBuffer<f32>,
    m: usize,
    k: usize,
    beta: f32,
    grid: u32,
) -> Result<(), Box<dyn Error>> {
    let mut a_ptr: CUdeviceptr = a.cu_deviceptr();
    let mut a_len: usize = a.len();
    let mut x_ptr: CUdeviceptr = x.cu_deviceptr();
    let mut x_len: usize = x.len();
    let mut y_ptr: CUdeviceptr = y.cu_deviceptr();
    let mut m_v: usize = m;
    let mut k_v: usize = k;
    let mut beta_v: f32 = beta;
    let mut params: [*mut c_void; 8] = [
        &mut a_ptr as *mut _ as *mut c_void,
        &mut a_len as *mut _ as *mut c_void,
        &mut x_ptr as *mut _ as *mut c_void,
        &mut x_len as *mut _ as *mut c_void,
        &mut y_ptr as *mut _ as *mut c_void,
        &mut m_v as *mut _ as *mut c_void,
        &mut k_v as *mut _ as *mut c_void,
        &mut beta_v as *mut _ as *mut c_void,
    ];
    unsafe {
        cuda_core::simt::launch_kernel_on_stream(
            func,
            (grid, 1, 1),
            (256, 1, 1),
            0,
            stream,
            &mut params,
        )?;
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let ctx = CudaContext::new(0)?;
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx)?;

    // Optional ironsand PTX (path via IRONSAND_PTX). Without it we report
    // b-vs-a only and skip the A/B.
    let iron_func = match std::env::var("IRONSAND_PTX") {
        Ok(path) => {
            let ptx = std::fs::read_to_string(&path)?;
            let m = ctx.load_module_from_ptx_src(&ptx)?;
            let f = m.load_function("gemv_q4k_fast")?;
            println!("ironsand PTX loaded: {path}  (entry gemv_q4k_fast)\n");
            // Keep module alive via the function's Arc.
            Some(f)
        }
        Err(_) => {
            println!("IRONSAND_PTX unset — reporting cuda-oxide vs CPU only (no A/B).\n");
            None
        }
    };

    const WARMUP: usize = 20;
    const RUNS: usize = 64;

    // (rows=m, cols=k, run_correctness). k must be % 256 == 0.
    let shapes: &[(usize, usize, bool)] = &[
        (8, 512, true),        // zorro oracle shape (byte-identical)
        (4096, 4096, true),    // realistic decode shape (correctness too)
        (11008, 4096, false),  // down-proj shape
        (4096, 11008, false),  // up/gate-proj shape
    ];

    println!("== CORRECTNESS (same tolerance the ironsand kernel meets vs CPU: max-abs < 1e-2) ==");
    println!(
        "{:>6} {:>6} | {:>12} {:>12} | {:>12} {:>12}",
        "m", "k", "b-vs-a abs", "b-vs-a relL2", "b-vs-c abs", "b-vs-c relL2"
    );

    // Hold perf rows to print after the correctness block.
    let mut perf_rows: Vec<String> = Vec::new();

    for &(m, k, do_corr) in shapes {
        let w = build_weights(m, k);
        let x = build_x(k);
        let grid = (m as u32).div_ceil(8);
        let weight_bytes = (m * (k / 256) * BLK) as f64;

        let a_dev = DeviceBuffer::from_host(&stream, &w)?;
        let x_dev = DeviceBuffer::from_host(&stream, &x)?;

        // (b) cuda-oxide into a fresh zeroed y.
        let mut y_ox = DeviceBuffer::<f32>::zeroed(&stream, m)?;
        // for_num_elems(m*32) => block=256, grid=ceil(m*32/256)=m.div_ceil(8):
        // exactly ironsand's warp-per-row launch (8 warps/block, 1 warp/row).
        let cfg = LaunchConfig::for_num_elems((m * 32) as u32);
        unsafe {
            module.gemv_q4k_fast(&stream, cfg, &a_dev, &x_dev, &mut y_ox, m, k, 0.0f32)?;
        }
        stream.synchronize()?;
        let got_b = y_ox.to_host_vec(&stream)?;

        // (c) ironsand PTX, if present.
        let got_c = if let Some(f) = &iron_func {
            let y_ir = DeviceBuffer::<f32>::zeroed(&stream, m)?;
            unsafe { launch_ironsand(f, &stream, &a_dev, &x_dev, &y_ir, m, k, 0.0, grid)? };
            stream.synchronize()?;
            Some(y_ir.to_host_vec(&stream)?)
        } else {
            None
        };

        if do_corr {
            let deq = cpu_dequant_q4k(&w, m * k);
            let a_ref = cpu_gemv(&deq, &x, m, k);
            let (ba_abs, ba_rel) = err(&got_b, &a_ref);
            let (bc_abs, bc_rel) = match &got_c {
                Some(c) => err(&got_b, c),
                None => (f32::NAN, f32::NAN),
            };
            println!(
                "{m:>6} {k:>6} | {ba_abs:>12.3e} {ba_rel:>12.3e} | {bc_abs:>12.3e} {bc_rel:>12.3e}"
            );
        }

        // ---- Perf: matched A/B (alternate b and c within one loop) ----
        let mut launch_ox = || -> Result<(), Box<dyn Error>> {
            unsafe {
                module.gemv_q4k_fast(&stream, cfg, &a_dev, &x_dev, &mut y_ox, m, k, 0.0f32)?;
            }
            Ok(())
        };
        for _ in 0..WARMUP {
            launch_ox()?;
        }
        stream.synchronize()?;

        let mut t_ox: Vec<f32> = Vec::with_capacity(RUNS);
        let mut t_ir: Vec<f32> = Vec::with_capacity(RUNS);
        let y_ir_perf = if iron_func.is_some() {
            Some(DeviceBuffer::<f32>::zeroed(&stream, m)?)
        } else {
            None
        };
        if let Some(f) = &iron_func {
            let y = y_ir_perf.as_ref().unwrap();
            for _ in 0..WARMUP {
                unsafe { launch_ironsand(f, &stream, &a_dev, &x_dev, y, m, k, 0.0, grid)? };
            }
            stream.synchronize()?;
        }

        for _ in 0..RUNS {
            // (b)
            let s = stream.record_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            launch_ox()?;
            let e = stream.record_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))?;
            e.synchronize()?;
            t_ox.push(s.elapsed_ms(&e)?);
            // (c)
            if let Some(f) = &iron_func {
                let y = y_ir_perf.as_ref().unwrap();
                let s = stream.record_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))?;
                unsafe { launch_ironsand(f, &stream, &a_dev, &x_dev, y, m, k, 0.0, grid)? };
                let e = stream.record_event(Some(CUevent_flags_enum_CU_EVENT_DEFAULT))?;
                e.synchronize()?;
                t_ir.push(s.elapsed_ms(&e)?);
            }
        }

        let ox_ms = median(t_ox);
        let ox_us = ox_ms * 1e3;
        let ox_gbs = weight_bytes / (ox_ms as f64 / 1e3) / 1e9;
        if let Some(f) = &iron_func {
            let _ = f;
            let ir_ms = median(t_ir);
            let ir_us = ir_ms * 1e3;
            let ir_gbs = weight_bytes / (ir_ms as f64 / 1e3) / 1e9;
            perf_rows.push(format!(
                "{m:>6} {k:>6} | {ox_us:>10.1} {ox_gbs:>9.1} | {ir_us:>10.1} {ir_gbs:>9.1} | {:>6.3}x",
                ir_ms / ox_ms
            ));
        } else {
            perf_rows.push(format!(
                "{m:>6} {k:>6} | {ox_us:>10.1} {ox_gbs:>9.1} |          -         - |      -"
            ));
        }
    }

    println!("\n== PERF: matched A/B, CUDA events, {WARMUP} warmup + {RUNS} runs, median ==");
    println!("(GB/s = weight_bytes / time; weights dominate GEMV traffic)");
    println!(
        "{:>6} {:>6} | {:>10} {:>9} | {:>10} {:>9} | {:>6}",
        "m", "k", "oxide us", "oxGB/s", "iron us", "irGB/s", "iron/ox"
    );
    for r in perf_rows {
        println!("{r}");
    }

    Ok(())
}
