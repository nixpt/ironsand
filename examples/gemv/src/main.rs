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

/// Stage-0 spike: one int8 `m16n8k32` tensor-core mma tile vs a CPU int8 reference.
/// Proves `rustc_codegen_nvvm`/LLVM19 can emit `mma.sync.*.s8` inline asm correctly.
fn run_mma_spike(module: &Module, stream: &Stream) -> Result<(), Box<dyn Error>> {
    use cust::memory::CopyDestination as _;
    const M: usize = 16;
    const N: usize = 8;
    const K: usize = 32;

    // Deterministic int8 fill in [-3, 4] (xorshift) — small so sums stay readable.
    let mut s: u32 = 0x1234_5678;
    let mut nxt = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        ((s >> 3) & 7) as i32 as i8 - 3
    };
    let a: Vec<i8> = (0..M * K).map(|_| nxt()).collect();
    let b: Vec<i8> = (0..N * K).map(|_| nxt()).collect();

    // CPU reference: C[i][j] = Σ_k A[i][k] · B[j][k].
    let mut c_ref = vec![0i32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0i32;
            for k in 0..K {
                acc += a[i * K + k] as i32 * b[j * K + k] as i32;
            }
            c_ref[i * N + j] = acc;
        }
    }

    let a_u8: Vec<u8> = a.iter().map(|&v| v as u8).collect();
    let b_u8: Vec<u8> = b.iter().map(|&v| v as u8).collect();
    let a_gpu = a_u8.as_slice().as_dbuf()?;
    let b_gpu = b_u8.as_slice().as_dbuf()?;
    let c_gpu = vec![0i32; M * N].as_slice().as_dbuf()?;

    let f = module.get_function("mma_int8_tile")?;
    unsafe {
        launch!(f<<<1, 32, 0, stream>>>(
            a_gpu.as_device_ptr(), a_gpu.len(),
            b_gpu.as_device_ptr(), b_gpu.len(),
            c_gpu.as_device_ptr()
        ))?;
    }
    stream.synchronize()?;

    let mut c_out = vec![0i32; M * N];
    c_gpu.copy_to(&mut c_out)?;

    let mut bad = 0usize;
    for i in 0..M * N {
        if c_out[i] != c_ref[i] {
            if bad < 8 {
                println!("  mismatch [{},{}] gpu={} cpu={}", i / N, i % N, c_out[i], c_ref[i]);
            }
            bad += 1;
        }
    }
    if bad == 0 {
        println!(
            "MMA-SPIKE: PASS — mma.sync.m16n8k32.s8 emits through nvvm/LLVM19 + correct \
             ({M}x{N}x{K} int8 tile, all {} elems match CPU)",
            M * N
        );
    } else {
        println!("MMA-SPIKE: FAIL — {bad}/{} elems mismatch (mma emitted but wrong layout/codegen)", M * N);
    }
    Ok(())
}

/// Stage-1: fused Q4_K int8 mmq batched GEMM correctness vs an f64 reference.
/// `Y[N×M] = X[N×K] · dequant(W)ᵀ`, W=Q4_K. Validates the fused in-kernel
/// act-quant + dequant + `d·sc`/`dmin·mn` correction + `[N×M]` tiling. The only
/// error vs the reference is the per-token int8 activation quant (ref uses the
/// dequantized weights, so weight-quant error is already folded in).
fn run_q4k_mmq(module: &Module, stream: &Stream) -> Result<(), Box<dyn Error>> {
    use cust::memory::CopyDestination as _;
    let (n, mrows, k) = (512usize, 2048usize, 2048usize); // prefill: N tokens × cols × rows

    let w = Array2::<f32>::random((mrows, k), Uniform::new(-1.0, 1.0));
    let (wq, w_deq) = quantize_q4k(&w, mrows, k);
    let xa = Array2::<f32>::random((n, k), Uniform::new(-1.0, 1.0));

    // Reference Y[n×m] = X · dequant(W)ᵀ in f64.
    let wd = Array2::from_shape_vec((mrows, k), w_deq).unwrap().mapv(|v| v as f64);
    let y_ref = xa.mapv(|v| v as f64).dot(&wd.t());

    let wq_gpu = wq.as_slice().as_dbuf()?;
    let x_flat: Vec<f32> = xa.as_standard_layout().iter().copied().collect();
    let x_gpu = x_flat.as_slice().as_dbuf()?;
    let y_gpu = vec![0.0f32; n * mrows].as_slice().as_dbuf()?;

    let f = module.get_function("gemm_q4k_mmq_dp4a")?;
    let run = || -> Result<(), Box<dyn Error>> {
        unsafe {
            launch!(f<<<n as u32, 256, 0, stream>>>(
                wq_gpu.as_device_ptr(), wq_gpu.len(),
                x_gpu.as_device_ptr(), x_gpu.len(),
                y_gpu.as_device_ptr(), n, mrows, k
            ))?;
        }
        Ok(())
    };
    let ms = time(stream, NUM_WARMUPS, NUM_RUNS, run)?;

    let mut y_out = vec![0.0f32; n * mrows];
    y_gpu.copy_to(&mut y_out)?;

    // Relative L2-norm error = the standard GEMM correctness measure. Per-element
    // rel error explodes on near-zero (cancellation) outputs even when the kernel
    // is exact — so judge on L2, and separately report rel error on the
    // large-magnitude outputs (|y| > 0.25·std) where a real bug would show.
    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut sum_sq = 0.0f64;
    for idx in 0..n * mrows {
        let r = y_ref[[idx / mrows, idx % mrows]];
        sum_sq += r * r;
    }
    let std = (sum_sq / (n * mrows) as f64).sqrt();
    let big_floor = 0.25 * std;
    let mut max_rel_big = 0.0f64;
    for idx in 0..n * mrows {
        let r = y_ref[[idx / mrows, idx % mrows]];
        let g = y_out[idx] as f64;
        let d = g - r;
        diff_sq += d * d;
        ref_sq += r * r;
        if r.abs() > big_floor {
            let rel = d.abs() / r.abs();
            if rel > max_rel_big {
                max_rel_big = rel;
            }
        }
    }
    let l2_rel = (diff_sq / ref_sq).sqrt();
    let tflops = 2.0 * n as f64 * mrows as f64 * k as f64 / (ms as f64 * 1e-3) / 1e12;
    println!(
        "Q4K-MMQ dp4a (fused, per-token int8 act): N={n} K={k} M={mrows}  {ms:.3} ms  {tflops:.1} TFLOP/s",
    );
    println!(
        "  L2-rel={l2_rel:.5}  max_rel(|y|>{big_floor:.2})={max_rel_big:.4}  (std(y)={std:.2})"
    );
    // L2-rel ~1e-2 = pure int8-act-quant noise (the math is exact); a layout/index
    // bug would blow L2-rel to ≫0.1 and corrupt the large-magnitude outputs too.
    if l2_rel < 0.02 && max_rel_big < 0.08 {
        println!("Q4K-MMQ: PASS — fused int8 mmq batched GEMM correct (dp4a inner). Stage 2 = swap → mma.sync + f16 out.");
    } else {
        println!("Q4K-MMQ: FAIL — L2-rel {l2_rel:.4} / max_rel_big {max_rel_big:.4} (structured error, not quant noise)");
    }
    Ok(())
}

/// Stage-2: tensor-core Q4_K mmq GEMM (`quant_act_q8` → `gemm_q4k_mma`, f16 out)
/// vs the f64 reference + vs the ~60 TFLOP/s cuBLAS-f16 bar.
fn run_q4k_mma(module: &Module, stream: &Stream) -> Result<(), Box<dyn Error>> {
    use cust::memory::CopyDestination as _;
    let (n, mrows, k) = (512usize, 2048usize, 2048usize);
    let nsub = k / 32;

    let w = Array2::<f32>::random((mrows, k), Uniform::new(-1.0, 1.0));
    let (wq, w_deq) = quantize_q4k(&w, mrows, k);
    let xa = Array2::<f32>::random((n, k), Uniform::new(-1.0, 1.0));
    let wd = Array2::from_shape_vec((mrows, k), w_deq).unwrap().mapv(|v| v as f64);
    let y_ref = xa.mapv(|v| v as f64).dot(&wd.t());

    let wq_gpu = wq.as_slice().as_dbuf()?;
    let x_flat: Vec<f32> = xa.as_standard_layout().iter().copied().collect();
    let x_gpu = x_flat.as_slice().as_dbuf()?;
    let xq_gpu = vec![0u8; n * k].as_slice().as_dbuf()?;
    let xscale_gpu = vec![0.0f32; n].as_slice().as_dbuf()?;
    let bsum_gpu = vec![0i32; n * nsub].as_slice().as_dbuf()?;
    let y_gpu = vec![0u16; n * mrows].as_slice().as_dbuf()?;

    let qf = module.get_function("quant_act_q8")?;
    let gf = module.get_function("gemm_q4k_mma")?;

    // Activation quant prologue (once); then time the gemm.
    let quant = || -> Result<(), Box<dyn Error>> {
        unsafe {
            launch!(qf<<<n as u32, 256, 0, stream>>>(
                x_gpu.as_device_ptr(), x_gpu.len(),
                xq_gpu.as_device_ptr(), xscale_gpu.as_device_ptr(), bsum_gpu.as_device_ptr(),
                n, k
            ))?;
        }
        Ok(())
    };
    quant()?;
    stream.synchronize()?;

    let grid = ((n / 16) * (mrows / 8)) as u32;
    let gemm = || -> Result<(), Box<dyn Error>> {
        unsafe {
            launch!(gf<<<grid, 32, 0, stream>>>(
                wq_gpu.as_device_ptr(), wq_gpu.len(),
                xq_gpu.as_device_ptr(), xq_gpu.len(),
                xscale_gpu.as_device_ptr(), xscale_gpu.len(),
                bsum_gpu.as_device_ptr(), bsum_gpu.len(),
                y_gpu.as_device_ptr(), n, mrows, k
            ))?;
        }
        Ok(())
    };
    let ms = time(stream, NUM_WARMUPS, NUM_RUNS, gemm)?;

    let mut y_bits = vec![0u16; n * mrows];
    y_gpu.copy_to(&mut y_bits)?;

    let mut diff_sq = 0.0f64;
    let mut ref_sq = 0.0f64;
    let mut sum_sq = 0.0f64;
    for idx in 0..n * mrows {
        let r = y_ref[[idx / mrows, idx % mrows]];
        sum_sq += r * r;
    }
    let std = (sum_sq / (n * mrows) as f64).sqrt();
    let big_floor = 0.25 * std;
    let mut max_rel_big = 0.0f64;
    for idx in 0..n * mrows {
        let r = y_ref[[idx / mrows, idx % mrows]];
        let g = half::f16::from_bits(y_bits[idx]).to_f32() as f64;
        let d = g - r;
        diff_sq += d * d;
        ref_sq += r * r;
        if r.abs() > big_floor {
            let rel = d.abs() / r.abs();
            if rel > max_rel_big {
                max_rel_big = rel;
            }
        }
    }
    let l2_rel = (diff_sq / ref_sq).sqrt();
    let tflops = 2.0 * n as f64 * mrows as f64 * k as f64 / (ms as f64 * 1e-3) / 1e12;
    println!(
        "Q4K-MMA tensor-core (fused, f16 out): N={n} K={k} M={mrows}  {ms:.4} ms  {tflops:.1} TFLOP/s  (bar: cuBLAS-f16 ~60)"
    );
    println!("  L2-rel={l2_rel:.5}  max_rel(|y|>{big_floor:.2})={max_rel_big:.4}  (std(y)={std:.2})");
    // f16 out adds ~5e-4 rounding on top of int8-act noise; tolerate a hair more.
    if l2_rel < 0.025 && max_rel_big < 0.10 {
        println!("Q4K-MMA: PASS — int8 tensor-core mmq GEMM correct.");
    } else {
        println!("Q4K-MMA: FAIL — L2-rel {l2_rel:.4} / max_rel_big {max_rel_big:.4} (fragment/scale bug)");
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
    let mut cublas = CublasContext::new()?;

    // Stage-0 spike: int8 tensor-core mma.sync go/no-go (ZORRO_MMA_SPIKE=1).
    if std::env::var("ZORRO_MMA_SPIKE").is_ok() {
        return run_mma_spike(&module, &stream);
    }
    // Stage-1: fused Q4_K int8 mmq batched GEMM correctness (ZORRO_Q4K_MMQ=1).
    if std::env::var("ZORRO_Q4K_MMQ").is_ok() {
        return run_q4k_mmq(&module, &stream);
    }
    // Stage-2: tensor-core (mma.sync) Q4_K mmq GEMM, f16 out (ZORRO_Q4K_MMA=1).
    if std::env::var("ZORRO_Q4K_MMA").is_ok() {
        return run_q4k_mma(&module, &stream);
    }

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

        // --- Q4_K (GGUF 4-bit k-quant, 4.5 bits/weight) -----------------------
        // Q4_K super-blocks are 256 weights; only applicable when k % 256 == 0.
        if k % 256 == 0 {
        let (q4k_blocks, a_deq) = quantize_q4k(&a, m, k);
        let ad_q4k = ndarray::Array2::from_shape_vec((m, k), a_deq).unwrap().mapv(|v| v as f64);
        let y_ref_q4k: Array1<f64> = ad_q4k.dot(&x.mapv(|v| v as f64));
        // int8-activation reference for the W4A8 vecdot kernel (acts = sx * x_q8).
        let x_a8_q4k: Array1<f64> = x_q8.iter().map(|&q| sx as f64 * (q as i8 as f64)).collect();
        let y_ref_q4k_a8: Array1<f64> = ad_q4k.dot(&x_a8_q4k);
        let q4k_gpu = q4k_blocks.as_slice().as_dbuf()?;
        let q4k_bytes = (m * (k / 256) * 144) as f64; // real Q4_K storage = m*k*0.5625
        stream.synchronize()?;

        let q4k = module.get_function("gemv_q4k_warp")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q4k<<<grid, block, 0, stream>>>(
                    q4k_gpu.as_device_ptr(), q4k_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q4_K", ms, q4k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q4k, 0.02)?);

        // optimized: lane owns whole sub-blocks (header decode amortized 32x)
        let q4kf = module.get_function("gemv_q4k_fast")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q4kf<<<grid, block, 0, stream>>>(
                    q4k_gpu.as_device_ptr(), q4k_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q4_K fast", ms, q4k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q4k, 0.02)?);

        // optimized v3: pair-of-sub-blocks + u32 scale reads + FMA
        let q4kv3 = module.get_function("gemv_q4k_v3")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q4kv3<<<grid, block, 0, stream>>>(
                    q4k_gpu.as_device_ptr(), q4k_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q4_K v3", ms, q4k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q4k, 0.02)?);

        // v4: 2-way super-block unroll on top of v3.
        let q4kv4 = module.get_function("gemv_q4k_v4")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q4kv4<<<grid, block, 0, stream>>>(
                    q4k_gpu.as_device_ptr(), q4k_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q4_K v4", ms, q4k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q4k, 0.02)?);

        // coalesced mmvq-style vec_dot (W4A8) — the mmvq-beating candidate
        let q4kv = module.get_function("gemv_q4k_vecdot")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q4kv<<<grid, block, 0, stream>>>(
                    q4k_gpu.as_device_ptr(), q4k_gpu.len(),
                    xq_gpu.as_device_ptr(), xq_gpu.len(),
                    sx, y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q4_K vecdot", ms, q4k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q4k_a8, 0.03)?);

        // --- Q6_K (GGUF 6-bit k-quant, the lm_head format) ------------------
        let (q6k_blocks, a6_deq) = quantize_q6k(&a, m, k);
        let a6 = ndarray::Array2::from_shape_vec((m, k), a6_deq).unwrap();
        let y_ref_q6k: Array1<f64> = a6.mapv(|v| v as f64).dot(&x.mapv(|v| v as f64));
        // int8-activation reference for the W6A8 kernel (acts = sx * x_q8).
        let x_a8: Array1<f64> = x_q8.iter().map(|&q| sx as f64 * (q as i8 as f64)).collect();
        let y_ref_q6k_a8: Array1<f64> = a6.mapv(|v| v as f64).dot(&x_a8);
        let q6k_gpu = q6k_blocks.as_slice().as_dbuf()?;
        let q6k_bytes = (m * (k / 256) * 210) as f64; // 6.5625 bits/weight
        stream.synchronize()?;
        let q6k = module.get_function("gemv_q6k_warp")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q6k<<<grid, block, 0, stream>>>(
                    q6k_gpu.as_device_ptr(), q6k_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q6_K W6A32", ms, q6k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q6k, 0.02)?);

        // Optimized Q6_K: mul_add FMA + 2-way super-block unroll.
        let q6kf = module.get_function("gemv_q6k_fast")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q6kf<<<grid, block, 0, stream>>>(
                    q6k_gpu.as_device_ptr(), q6k_gpu.len(),
                    x_gpu.as_device_ptr(), x_gpu.len(),
                    y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q6_K fast", ms, q6k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q6k, 0.02)?);

        // W6A8: int8 activations + dp4a integer dot (the mmvq-style path)
        let q6kd = module.get_function("gemv_q6k_dp4a")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q6kd<<<grid, block, 0, stream>>>(
                    q6k_gpu.as_device_ptr(), q6k_gpu.len(),
                    xq_gpu.as_device_ptr(), xq_gpu.len(),
                    sx, y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q6_K W6A8", ms, q6k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q6k_a8, 0.03)?);

        // coalesced mmvq-style vec_dot (W6A8)
        let q6kv = module.get_function("gemv_q6k_vecdot")?;
        let run = || -> Result<(), Box<dyn Error>> {
            let block = 256u32;
            let grid = (m as u32).div_ceil(block / 32);
            unsafe {
                launch!(q6kv<<<grid, block, 0, stream>>>(
                    q6k_gpu.as_device_ptr(), q6k_gpu.len(),
                    xq_gpu.as_device_ptr(), xq_gpu.len(),
                    sx, y_gpu.as_device_ptr(), m, k, beta
                ))?;
            }
            Ok(())
        };
        let ms = time(&stream, NUM_WARMUPS, NUM_RUNS, run)?;
        report(&label, "Q6_K vecdot", ms, q6k_bytes, &check_eps(&stream, &mut y_gpu, m, &y_ref_q6k_a8, 0.03)?);
        }

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

/// Quantize a row-major `m x k` f32 matrix to byte-faithful Q4_K super-blocks
/// (`k % 256 == 0`). Returns the packed blocks and the dequantized weights
/// (what the format represents) for the GEMV reference.
fn quantize_q4k(a: &Array2<f32>, m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let nb = k / 256;
    let mut blocks = vec![0u8; m * nb * 144];
    let mut a_deq = vec![0.0f32; m * k];
    for i in 0..m {
        for bk in 0..nb {
            let wbase = bk * 256;
            let mut sub_scale = [0.0f32; 8];
            let mut sub_off = [0.0f32; 8];
            let mut q = [0u8; 256];
            for sb in 0..8 {
                let (mut mn, mut mx) = (f32::INFINITY, f32::NEG_INFINITY);
                for l in 0..32 {
                    let v = a[[i, wbase + sb * 32 + l]];
                    mn = mn.min(v);
                    mx = mx.max(v);
                }
                let sc = ((mx - mn) / 15.0).max(1e-8);
                sub_scale[sb] = sc;
                sub_off[sb] = (-mn).max(0.0); // affine offset (data here has mn<0)
                for l in 0..32 {
                    let qq = (((a[[i, wbase + sb * 32 + l]] - mn) / sc).round()).clamp(0.0, 15.0);
                    q[sb * 32 + l] = qq as u8;
                }
            }
            let d = (sub_scale.iter().cloned().fold(0.0f32, f32::max).max(1e-8)) / 63.0;
            let doff = (sub_off.iter().cloned().fold(0.0f32, f32::max).max(1e-8)) / 63.0;
            let mut sc6 = [0u8; 8];
            let mut mn6 = [0u8; 8];
            for sb in 0..8 {
                sc6[sb] = (sub_scale[sb] / d).round().clamp(0.0, 63.0) as u8;
                mn6[sb] = (sub_off[sb] / doff).round().clamp(0.0, 63.0) as u8;
            }
            let base = (i * nb + bk) * 144;
            let dh = half::f16::from_f32(d).to_bits();
            let dmh = half::f16::from_f32(doff).to_bits();
            blocks[base] = dh as u8;
            blocks[base + 1] = (dh >> 8) as u8;
            blocks[base + 2] = dmh as u8;
            blocks[base + 3] = (dmh >> 8) as u8;
            // pack scales[12] (inverse of llama.cpp get_scale_min_k4)
            let sb0 = base + 4;
            for j in 0..4 {
                blocks[sb0 + j] = (sc6[j] & 63) | ((sc6[j + 4] >> 4) << 6);
                blocks[sb0 + j + 4] = (mn6[j] & 63) | ((mn6[j + 4] >> 4) << 6);
                blocks[sb0 + j + 8] = (sc6[j + 4] & 0xF) | ((mn6[j + 4] & 0xF) << 4);
            }
            // pack qs[128]: group g low nibble = sub 2g, high nibble = sub 2g+1
            let qb = base + 16;
            for g in 0..4 {
                for l in 0..32 {
                    blocks[qb + g * 32 + l] = q[2 * g * 32 + l] | (q[(2 * g + 1) * 32 + l] << 4);
                }
            }
            // dequant reference (f16-rounded d/dmin, as the kernel reads them)
            let df = half::f16::from_f32(d).to_f32();
            let dmf = half::f16::from_f32(doff).to_f32();
            for sb in 0..8 {
                let d_eff = df * sc6[sb] as f32;
                let m_eff = dmf * mn6[sb] as f32;
                for l in 0..32 {
                    a_deq[i * k + wbase + sb * 32 + l] =
                        d_eff * (q[sb * 32 + l] as f32) - m_eff;
                }
            }
        }
    }
    (blocks, a_deq)
}

/// Quantize a row-major `m x k` f32 matrix to byte-faithful Q6_K super-blocks
/// (`k % 256 == 0`). Returns the packed blocks and the dequantized weights.
fn quantize_q6k(a: &Array2<f32>, m: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let nb = k / 256;
    let mut blocks = vec![0u8; m * nb * 210];
    let mut a_deq = vec![0.0f32; m * k];
    for i in 0..m {
        for bk in 0..nb {
            let wbase = bk * 256;
            let mut scale_sb = [0.0f32; 16];
            let mut q6 = [0u8; 256];
            for sb in 0..16 {
                let mut amax = 0.0f32;
                for l in 0..16 {
                    amax = amax.max(a[[i, wbase + sb * 16 + l]].abs());
                }
                let s = (amax / 32.0).max(1e-8);
                scale_sb[sb] = s;
                for l in 0..16 {
                    let q = ((a[[i, wbase + sb * 16 + l]] / s).round() + 32.0).clamp(0.0, 63.0);
                    q6[sb * 16 + l] = q as u8;
                }
            }
            let d = (scale_sb.iter().cloned().fold(0.0f32, f32::max).max(1e-8)) / 127.0;
            let mut sc8 = [0i8; 16];
            for sb in 0..16 {
                sc8[sb] = (scale_sb[sb] / d).round().clamp(-127.0, 127.0) as i8;
            }
            let base = (i * nb + bk) * 210;
            // pack ql (low nibble) + qh (2 high bits) per llama.cpp interleaving
            for p in 0..256 {
                let group = p / 128;
                let pos = p % 128;
                let q = q6[p];
                let (lo, hi) = (q & 0xF, q >> 4);
                let (ql_off, ql_high, qh_off, qh_shift) = match pos / 32 {
                    0 => (group * 64 + pos, false, group * 32 + pos, 0u8),
                    1 => (group * 64 + (pos - 32) + 32, false, group * 32 + (pos - 32), 2),
                    2 => (group * 64 + (pos - 64), true, group * 32 + (pos - 64), 4),
                    _ => (group * 64 + (pos - 96) + 32, true, group * 32 + (pos - 96), 6),
                };
                if ql_high {
                    blocks[base + ql_off] |= lo << 4;
                } else {
                    blocks[base + ql_off] |= lo;
                }
                blocks[base + 128 + qh_off] |= hi << qh_shift;
            }
            for sb in 0..16 {
                blocks[base + 192 + sb] = sc8[sb] as u8;
            }
            let dh = half::f16::from_f32(d).to_bits();
            blocks[base + 208] = dh as u8;
            blocks[base + 209] = (dh >> 8) as u8;
            // dequant reference
            let df = half::f16::from_f32(d).to_f32();
            for p in 0..256 {
                let sb = p / 16;
                a_deq[i * k + wbase + p] = df * (sc8[sb] as f32) * (q6[p] as f32 - 32.0);
            }
        }
    }
    (blocks, a_deq)
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
