//! GEMV benchmark: hand-written Rust kernels vs the cuBLAS N=1 GEMM baseline.
//!
//! GEMV (`y = A·x`) is the decode-phase hot path of LLM inference: every weight
//! projection with a batch/sequence of 1 is a matrix-vector product. It is
//! memory-bandwidth bound — the time is dominated by streaming the `m*k` weight
//! matrix `A` once — so we report effective A-read bandwidth (GB/s) alongside
//! latency. cuBLAS has no dedicated GEMV here (blastoff exposes only GEMM), and
//! N=1 GEMM is exactly the floor zorro's decode path hits, so it is the right
//! baseline to beat (or match).

use std::error::Error;

use blastoff::{CublasContext, MatrixOp};
use cust::event::{Event, EventFlags};
use cust::launch;
use cust::memory::{CopyDestination as _, DeviceBox, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt as _;
use ndarray::{Array1, Array2};
use ndarray_rand::RandomExt as _;
use ndarray_rand::rand_distr::Uniform;

const NUM_WARMUPS: usize = 3;
const NUM_RUNS: usize = 50;
const EPS: f32 = 0.02;

/// (m, k) = (out_features, in_features). Decode-shaped projections: attention
/// q/k/v/o and MLP gate/up/down for a ~7B model, plus the vocab lm_head, plus a
/// small shape for a quick correctness anchor.
const SHAPES: [(usize, usize); 6] = [
    (256, 384),     // correctness anchor
    (4096, 4096),   // attn projection
    (11008, 4096),  // mlp gate/up
    (4096, 11008),  // mlp down
    (12288, 4096),  // fused qkv
    (32000, 4096),  // lm_head (vocab)
];

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn main() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut cublas = CublasContext::new()?;

    println!(
        "{:>12} {:>10} {:>11} {:>11}   {:>9}",
        "shape (m x k)", "method", "ms", "GB/s", "result"
    );
    println!("{}", "-".repeat(64));

    let (alpha, beta) = (1.0f32, 0.0f32);

    for (m, k) in SHAPES {
        // Host data.
        let a = Array2::<f32>::random((m, k), Uniform::new(-1.0, 1.0));
        let x = Array1::<f32>::random(k, Uniform::new(-1.0, 1.0));
        // CPU reference in f64 (beta = 0, so initial y is irrelevant).
        let y_ref: Array1<f64> = {
            let a64 = a.mapv(|v| v as f64);
            let x64 = x.mapv(|v| v as f64);
            a64.dot(&x64) * alpha as f64
        };

        // Device buffers.
        let a_gpu = a.as_standard_layout().as_slice().unwrap().as_dbuf()?;
        let x_gpu = x.as_slice().unwrap().as_dbuf()?;
        let mut y_gpu = vec![0.0f32; m].as_slice().as_dbuf()?;
        let alpha_gpu = DeviceBox::new(&alpha)?;
        let beta_gpu = DeviceBox::new(&beta)?;
        stream.synchronize()?;

        let a_bytes = (m * k * std::mem::size_of::<f32>()) as f64;
        let label = format!("{m}x{k}");

        // --- cuBLAS N=1 GEMM baseline -----------------------------------------
        // Row-major A (m x k) is column-major Aᵀ (k x m, ld=k); op=Transpose
        // recovers A, so C(m x 1) = A·x.  C ldc = m.
        let cublas_run = |cublas: &mut CublasContext, y: &mut DeviceBuffer<f32>| -> Result<(), Box<dyn Error>> {
            cublas.gemm::<f32>(
                &stream, m, 1, k,
                &alpha_gpu, &a_gpu, k, MatrixOp::Transpose,
                &beta_gpu, &x_gpu, k, MatrixOp::None,
                y, m,
            )?;
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, || cublas_run(&mut cublas, &mut y_gpu))?;
        report(&label, "cuBLAS(N=1)", ms, a_bytes, &check(&stream, &mut y_gpu, m, &y_ref)?);

        // --- Rust naive: thread per row ---------------------------------------
        let naive = module.get_function("gemv_naive")?;
        let naive_run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block);
            unsafe {
                launch!(naive<<<grid, block, 0, stream>>>(
                    a_gpu.as_device_ptr(), a_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, alpha, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, naive_run)?;
        report(&label, "rust naive", ms, a_bytes, &check(&stream, &mut y_gpu, m, &y_ref)?);

        // --- Rust block: block per row, shared-memory reduction ---------------
        let block_k = module.get_function("gemv_block")?;
        let block_run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32; // must match BLOCK in the kernel
            let grid = m as u32; // one block per row
            unsafe {
                launch!(block_k<<<grid, block, 0, stream>>>(
                    a_gpu.as_device_ptr(), a_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, alpha, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, block_run)?;
        report(&label, "rust block", ms, a_bytes, &check(&stream, &mut y_gpu, m, &y_ref)?);

        // --- Rust warp: warp per row, shuffle reduction -----------------------
        let warp = module.get_function("gemv_warp")?;
        let warp_run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32; // 8 warps per block
            let warps_per_block = block / 32;
            let grid = (m as u32).div_ceil(warps_per_block);
            unsafe {
                launch!(warp<<<grid, block, 0, stream>>>(
                    a_gpu.as_device_ptr(), a_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, alpha, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, warp_run)?;
        report(&label, "rust warp", ms, a_bytes, &check(&stream, &mut y_gpu, m, &y_ref)?);

        // --- f16 weights (the inference case: half the bytes) -----------------
        // Round A to f16, upload as raw u16 bits, and compute an f16-rounded
        // reference so correctness reflects what the f16 kernels should produce.
        let a_f16_bits: Vec<u16> = a
            .as_standard_layout()
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        let y_ref_f16: Array1<f64> = {
            let a_rounded = a.mapv(|v| half::f16::from_f32(v).to_f32() as f64);
            let x64 = x.mapv(|v| v as f64);
            a_rounded.dot(&x64) * alpha as f64
        };
        let a16_gpu = a_f16_bits.as_slice().as_dbuf()?;
        let f16_bytes = (m * k * std::mem::size_of::<u16>()) as f64;
        stream.synchronize()?;

        // cuBLAS f16 (hgemm) N=1 — the fair f16 vendor baseline.
        {
            let a_f16: Vec<half::f16> =
                a.as_standard_layout().iter().map(|&v| half::f16::from_f32(v)).collect();
            let x_f16: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
            let a16f = a_f16.as_slice().as_dbuf()?;
            let x16f = x_f16.as_slice().as_dbuf()?;
            let mut y16f = vec![half::f16::ZERO; m].as_slice().as_dbuf()?;
            let alpha16 = DeviceBox::new(&half::f16::from_f32(alpha))?;
            let beta16 = DeviceBox::new(&half::f16::from_f32(beta))?;
            let run = |cublas: &mut CublasContext, y: &mut DeviceBuffer<half::f16>| -> Result<(), Box<dyn Error>> {
                cublas.gemm::<half::f16>(
                    &stream, m, 1, k,
                    &alpha16, &a16f, k, MatrixOp::Transpose,
                    &beta16, &x16f, k, MatrixOp::None,
                    y, m,
                )?;
                Ok(())
            };
            let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, || run(&mut cublas, &mut y16f))?;
            stream.synchronize()?;
            let mut host = vec![half::f16::ZERO; m];
            y16f.copy_to(&mut host)?;
            let mut max_rel = 0.0f64;
            for (i, &got) in host.iter().enumerate() {
                let want = y_ref_f16[i];
                max_rel = max_rel.max(((got.to_f32() as f64) - want).abs() / want.abs().max(1.0));
            }
            // f16 accumulation (cublasHgemm) is lossy over large k; allow a wider band.
            let res = if max_rel <= 0.10 { "ok".to_string() } else { format!("FAIL ({max_rel:.3})") };
            report(&label, "cuBLAS f16", ms, f16_bytes, &res);
        }

        for (kname, label_k) in [("gemv_f16_warp", "f16 warp"), ("gemv_f16_vec4", "f16 vec4")] {
            let kf = module.get_function(kname)?;
            let run = || -> Result<(), Box<dyn Error>> {
                let block = 256u32; // 8 warps/block
                let grid = (m as u32).div_ceil(block / 32);
                unsafe {
                    launch!(kf<<<grid, block, 0, stream>>>(
                        a16_gpu.as_device_ptr(), a16_gpu.len(),
                        x_gpu.as_device_ptr(), x_gpu.len(),
                        y_gpu.as_device_ptr(), m, k, alpha, beta
                    ))?;
                }
                Ok(())
            };
            let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
            report(&label, label_k, ms, f16_bytes, &check(&stream, &mut y_gpu, m, &y_ref_f16)?);
        }

        // --- int8 weights (quantized decode: ¼ the bytes of f32) --------------
        // Per-row symmetric int8 quant for A; per-vector int8 quant for x.
        let mut a_q8 = vec![0u8; m * k];
        let mut scale_a = vec![0.0f32; m];
        for i in 0..m {
            let amax = (0..k).fold(0.0f32, |acc, j| acc.max(a[[i, j]].abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            scale_a[i] = s;
            for j in 0..k {
                a_q8[i * k + j] = ((a[[i, j]] / s).round().clamp(-127.0, 127.0) as i8) as u8;
            }
        }
        let xmax = x.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
        let sx = if xmax > 0.0 { xmax / 127.0 } else { 1.0 };
        let x_q8: Vec<u8> = x
            .iter()
            .map(|&v| ((v / sx).round().clamp(-127.0, 127.0) as i8) as u8)
            .collect();

        // References from the quantized data (f64).
        let mut y_ref_w8 = Array1::<f64>::zeros(m);
        let mut y_ref_dp4a = Array1::<f64>::zeros(m);
        for i in 0..m {
            let (mut s1, mut s2) = (0.0f64, 0i64);
            for j in 0..k {
                let q = a_q8[i * k + j] as i8 as f64;
                s1 += q * x[j] as f64;
                s2 += (a_q8[i * k + j] as i8 as i64) * (x_q8[j] as i8 as i64);
            }
            y_ref_w8[i] = scale_a[i] as f64 * s1;
            y_ref_dp4a[i] = scale_a[i] as f64 * sx as f64 * s2 as f64;
        }

        let aq_gpu = a_q8.as_slice().as_dbuf()?;
        let sa_gpu = scale_a.as_slice().as_dbuf()?;
        let xq_gpu = x_q8.as_slice().as_dbuf()?;
        let i8_bytes = (m * k) as f64;
        stream.synchronize()?;

        // W8A32 (int8 weights, f32 activations)
        let i8w = module.get_function("gemv_i8_warp")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(i8w<<<grid, block, 0, stream>>>(
                    aq_gpu.as_device_ptr(), aq_gpu.len(),
                    sa_gpu.as_device_ptr(), sa_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "i8 W8A32", ms, i8_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_w8, 0.03)?);

        // W8A8 via dp4a (int8 weights × int8 activations)
        let i8d = module.get_function("gemv_i8_dp4a")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(i8d<<<grid, block, 0, stream>>>(
                    aq_gpu.as_device_ptr(), aq_gpu.len(),
                    sa_gpu.as_device_ptr(), sa_gpu.len(),
                    xq_gpu.as_device_ptr(), xq_gpu.len(),
                    sx, y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "i8 dp4a", ms, i8_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_dp4a, 0.05)?);

        // --- ternary i2_s (BitNet: 2 bits/weight) -----------------------------
        // Per-row ternary quant, scale = mean(|row|); codes = w+1 ∈ {0,1,2}
        // packed 16 per u32. Reuses the int8 activations (x_q8, sx) from above.
        let kw = k / 16;
        let mut w_tern = vec![0u32; m * kw];
        let mut scale_w = vec![0.0f32; m];
        for i in 0..m {
            let mean_abs =
                (0..k).map(|j| a[[i, j]].abs() as f64).sum::<f64>() / k as f64;
            let s = (mean_abs as f32).max(1e-8);
            scale_w[i] = s;
            for j in 0..k {
                let t = (a[[i, j]] / s).round().clamp(-1.0, 1.0) as i32; // {-1,0,1}
                w_tern[i * kw + j / 16] |= ((t + 1) as u32) << ((j % 16) * 2);
            }
        }
        // Reference decoded from the packed words (also validates packing).
        let mut y_ref_tern = Array1::<f64>::zeros(m);
        for i in 0..m {
            let mut s = 0i64;
            for j in 0..k {
                let code = (w_tern[i * kw + j / 16] >> ((j % 16) * 2)) & 0x3;
                s += (code as i64 - 1) * (x_q8[j] as i8 as i64);
            }
            y_ref_tern[i] = scale_w[i] as f64 * sx as f64 * s as f64;
        }

        let wt_gpu = w_tern.as_slice().as_dbuf()?;
        let sw_gpu = scale_w.as_slice().as_dbuf()?;
        let tern_bytes = (m * k / 4) as f64; // 2 bits/weight
        stream.synchronize()?;

        let tk = module.get_function("gemv_ternary_warp")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(tk<<<grid, block, 0, stream>>>(
                    wt_gpu.as_device_ptr(), wt_gpu.len(),
                    sw_gpu.as_device_ptr(), sw_gpu.len(),
                    xq_gpu.as_device_ptr(), xq_gpu.len(),
                    sx, y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "ternary", ms, tern_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_tern, 0.02)?);

        // Optimized ternary: dp4a + branchless spread, with the Σx correction.
        let x_sum: i32 = x_q8.iter().map(|&b| b as i8 as i32).sum();
        let tkd = module.get_function("gemv_ternary_dp4a")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(tkd<<<grid, block, 0, stream>>>(
                    wt_gpu.as_device_ptr(), wt_gpu.len(),
                    sw_gpu.as_device_ptr(), sw_gpu.len(),
                    xq_gpu.as_device_ptr(), xq_gpu.len(),
                    sx, x_sum, y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "tern dp4a", ms, tern_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_tern, 0.02)?);

        println!();
    }

    Ok(())
}

/// Run `f` `warmups` times, then time `runs` back-to-back launches with events.
fn time<F>(stream: &Stream, warmups: usize, runs: usize, mut f: F) -> Result<f32, Box<dyn Error>>
where
    F: FnMut() -> Result<(), Box<dyn Error>>,
{
    for _ in 0..warmups {
        f()?;
    }
    stream.synchronize()?;
    let beg = Event::new(EventFlags::DEFAULT)?;
    let end = Event::new(EventFlags::DEFAULT)?;
    beg.record(stream)?;
    for _ in 0..runs {
        f()?;
    }
    end.record(stream)?;
    end.synchronize()?;
    Ok(end.elapsed_time_f32(&beg)? / runs as f32)
}

/// Copy `y_gpu` back and compare to the f64 reference; return "ok" / "FAIL".
fn check(
    stream: &Stream,
    y_gpu: &mut DeviceBuffer<f32>,
    m: usize,
    y_ref: &Array1<f64>,
) -> Result<String, Box<dyn Error>> {
    stream.synchronize()?;
    let mut host = vec![0.0f32; m];
    y_gpu.copy_to(&mut host)?;
    let mut max_rel = 0.0f64;
    for (i, &got) in host.iter().enumerate() {
        let want = y_ref[i];
        let denom = want.abs().max(1.0);
        max_rel = max_rel.max(((got as f64) - want).abs() / denom);
    }
    Ok(if max_rel <= EPS as f64 {
        "ok".to_string()
    } else {
        format!("FAIL ({max_rel:.3})")
    })
}

/// Like [`check`] but with an explicit relative-error tolerance (quantized
/// kernels carry more rounding than f32/f16).
fn check_eps(
    stream: &Stream,
    y_gpu: &mut DeviceBuffer<f32>,
    m: usize,
    y_ref: &Array1<f64>,
    eps: f64,
) -> Result<String, Box<dyn Error>> {
    stream.synchronize()?;
    let mut host = vec![0.0f32; m];
    y_gpu.copy_to(&mut host)?;
    let mut max_rel = 0.0f64;
    for (i, &got) in host.iter().enumerate() {
        let want = y_ref[i];
        max_rel = max_rel.max(((got as f64) - want).abs() / want.abs().max(1.0));
    }
    Ok(if max_rel <= eps {
        "ok".to_string()
    } else {
        format!("FAIL ({max_rel:.3})")
    })
}

fn report(label: &str, method: &str, ms: f32, a_bytes: f64, result: &str) {
    let gbps = a_bytes / (ms as f64 * 1.0e6);
    println!("{label:>12} {method:>10} {ms:>11.5} {gbps:>11.1}   {result:>9}");
}
