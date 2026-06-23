//! Attention kernel tests: f16 mma.sync spike + FlashAttention-2 correctness.

use std::error::Error;

use cust::event::{Event, EventFlags};
use cust::launch;
use cust::memory::{CopyDestination as _, DeviceBuffer};
use cust::module::Module;
use cust::stream::{Stream, StreamFlags};
use cust::util::SliceExt as _;
use half::f16;
use ndarray::Array2;
use ndarray_rand::RandomExt as _;
use ndarray_rand::rand_distr::Uniform;

const DH: usize = 128;
const NUM_WARMUPS: usize = 3;
const NUM_RUNS: usize = 50;

static PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernels.ptx"));

fn time<F: FnMut() -> Result<(), Box<dyn Error>>>(
    stream: &Stream,
    warmups: usize,
    runs: usize,
    mut f: F,
) -> Result<f32, Box<dyn Error>> {
    for _ in 0..warmups {
        f()?;
    }
    stream.synchronize()?;
    let start = Event::new(EventFlags::DEFAULT)?;
    let stop = Event::new(EventFlags::DEFAULT)?;
    start.record(stream)?;
    for _ in 0..runs {
        f()?;
    }
    stop.record(stream)?;
    stop.synchronize()?;
    Ok(stop.elapsed_time_f32(&start)? / runs as f32)
}

// ── F16 MMA spike ────────────────────────────────────────────────────────────

fn run_mma_f16_spike(module: &Module, stream: &Stream) -> Result<(), Box<dyn Error>> {
    const M: usize = 16;
    const N: usize = 8;
    const K: usize = 16;

    // Generate deterministic f16 test data (small integers for exact checking).
    // A [M=16, K=16] row-major f16; B [K=16, N=8] col-major f16 (stored as [N*K]).
    let mut s: u32 = 0x9E37_79B9;
    let mut nxt_f32 = || {
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        // Integers -2..2 (small, exact in f16 and f32).
        ((s >> 4) & 3) as i32 as f32 - 2.0
    };
    let a_f32: Vec<f32> = (0..M * K).map(|_| nxt_f32()).collect();
    let b_f32: Vec<f32> = (0..N * K).map(|_| nxt_f32()).collect();

    // CPU reference: D[i][j] = Σ_k A[i][k] * B_col[j*K + k].
    let mut d_ref = vec![0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = 0.0f32;
            for kk in 0..K {
                acc += a_f32[i * K + kk] * b_f32[j * K + kk];
            }
            d_ref[i * N + j] = acc;
        }
    }

    let a_h16: Vec<u16> = a_f32.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let b_h16: Vec<u16> = b_f32.iter().map(|&v| f16::from_f32(v).to_bits()).collect();
    let a_gpu = a_h16.as_slice().as_dbuf()?;
    let b_gpu = b_h16.as_slice().as_dbuf()?;
    let c_gpu = vec![0.0f32; M * N].as_slice().as_dbuf()?;

    let f = module.get_function("mma_f16_tile")?;
    unsafe {
        launch!(f<<<1, 32, 0, stream>>>(
            a_gpu.as_device_ptr(), a_gpu.len(),
            b_gpu.as_device_ptr(), b_gpu.len(),
            c_gpu.as_device_ptr()
        ))?;
    }
    stream.synchronize()?;

    let mut d_out = vec![0.0f32; M * N];
    c_gpu.copy_to(&mut d_out)?;

    let mut bad = 0usize;
    for i in 0..M * N {
        // f16 mma accumulates in f32 — values are exact for these small integers.
        if (d_out[i] - d_ref[i]).abs() > 0.01 {
            if bad < 4 {
                println!("  mismatch [{},{}] gpu={} cpu={}", i / N, i % N, d_out[i], d_ref[i]);
            }
            bad += 1;
        }
    }
    if bad == 0 {
        println!(
            "MMA-F16-SPIKE: PASS — mma.sync.m16n8k16.f16 emits through nvvm/LLVM19 + correct \
             ({M}×{N}×{K} f16→f32 tile, all {} elems match CPU)",
            M * N
        );
    } else {
        println!("MMA-F16-SPIKE: FAIL — {bad}/{} mismatches", M * N);
    }
    Ok(())
}

// ── Naive CPU attention reference ────────────────────────────────────────────

/// O[L, Dh] = softmax(Q[L,Dh] · K[S,Dh]^T / √Dh) · V[S, Dh]
fn naive_attn_cpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    l: usize,
    s: usize,
    dh: usize,
) -> Vec<f32> {
    let scale = 1.0 / (dh as f32).sqrt();
    let mut o = vec![0.0f32; l * dh];
    for i in 0..l {
        let mut scores = vec![0.0f32; s];
        for j in 0..s {
            let mut dot = 0.0f32;
            for d in 0..dh {
                dot += q[i * dh + d] * k[j * dh + d];
            }
            scores[j] = dot * scale;
        }
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let sum_e: f32 = scores.iter().map(|&v| (v - max_s).exp()).sum();
        let probs: Vec<f32> = scores.iter().map(|&v| (v - max_s).exp() / sum_e).collect();
        for d in 0..dh {
            let mut acc = 0.0f32;
            for j in 0..s {
                acc += probs[j] * v[j * dh + d];
            }
            o[i * dh + d] = acc;
        }
    }
    o
}

// ── Flash attention correctness + timing ─────────────────────────────────────

fn run_flash_attn(module: &Module, stream: &Stream) -> Result<(), Box<dyn Error>> {
    // Small shape for correctness, then a few larger ones for timing.
    let shapes: &[(usize, usize)] = &[
        (32, 32),    // smoke test
        (64, 64),    // still fast CPU ref
        (512, 512),  // first real shape
    ];

    for &(l_seq, s_seq) in shapes {
        run_flash_attn_shape(module, stream, l_seq, s_seq)?;
    }
    Ok(())
}

fn run_flash_attn_shape(
    module: &Module,
    stream: &Stream,
    l_seq: usize,
    s_seq: usize,
) -> Result<(), Box<dyn Error>> {
    assert!(l_seq % 16 == 0 && s_seq % 16 == 0, "L and S must be multiples of 16");

    // Generate f32 Q, K, V (attention-scale values: variance ~1/√Dh).
    let q_f = Array2::<f32>::random((l_seq, DH), Uniform::new(-1.0f32, 1.0));
    let k_f = Array2::<f32>::random((s_seq, DH), Uniform::new(-1.0f32, 1.0));
    let v_f = Array2::<f32>::random((s_seq, DH), Uniform::new(-0.5f32, 0.5));

    // CPU reference (f32 precision):
    let q_flat: Vec<f32> = q_f.as_standard_layout().iter().copied().collect();
    let k_flat: Vec<f32> = k_f.as_standard_layout().iter().copied().collect();
    let v_flat: Vec<f32> = v_f.as_standard_layout().iter().copied().collect();
    let o_ref = naive_attn_cpu(&q_flat, &k_flat, &v_flat, l_seq, s_seq, DH);

    // Convert to f16 for GPU:
    let to_h16 = |v: &[f32]| -> Vec<u16> { v.iter().map(|&x| f16::from_f32(x).to_bits()).collect() };
    let q_h16 = to_h16(&q_flat);
    let k_h16 = to_h16(&k_flat);
    let v_h16 = to_h16(&v_flat);

    let q_gpu = q_h16.as_slice().as_dbuf()?;
    let k_gpu = k_h16.as_slice().as_dbuf()?;
    let v_gpu = v_h16.as_slice().as_dbuf()?;
    let o_gpu = DeviceBuffer::<u16>::zeroed(l_seq * DH)?;

    let f = module.get_function("flash_attn")?;
    let grid = (l_seq / 16) as u32;

    let run = || -> Result<(), Box<dyn Error>> {
        unsafe {
            launch!(f<<<grid, 32, 0, stream>>>(
                q_gpu.as_device_ptr(), q_gpu.len(),
                k_gpu.as_device_ptr(), k_gpu.len(),
                v_gpu.as_device_ptr(), v_gpu.len(),
                o_gpu.as_device_ptr(),
                l_seq, s_seq
            ))?;
        }
        Ok(())
    };

    // Time it (skip for small shapes):
    let ms = if l_seq >= 512 {
        time(stream, NUM_WARMUPS, NUM_RUNS, run)?
    } else {
        run()?;
        stream.synchronize()?;
        0.0
    };

    let mut o_bits = vec![0u16; l_seq * DH];
    o_gpu.copy_to(&mut o_bits)?;
    let o_gpu_f: Vec<f32> = o_bits.iter().map(|&b| f16::from_bits(b).to_f32()).collect();

    // Relative L2 error.
    let diff_sq: f64 = o_ref
        .iter()
        .zip(&o_gpu_f)
        .map(|(&r, &g)| (r as f64 - g as f64).powi(2))
        .sum();
    let ref_sq: f64 = o_ref.iter().map(|&r| (r as f64).powi(2)).sum();
    let l2_rel = (diff_sq / ref_sq.max(1e-12)).sqrt();

    // Max absolute error on large-magnitude outputs (|o_ref| > 0.1).
    let mut max_abs_big = 0.0f64;
    for (&r, &g) in o_ref.iter().zip(&o_gpu_f) {
        if r.abs() > 0.1 {
            let e = (r as f64 - g as f64).abs();
            if e > max_abs_big {
                max_abs_big = e;
            }
        }
    }

    let timing = if ms > 0.0 {
        let flops = 4.0 * l_seq as f64 * s_seq as f64 * DH as f64; // QK^T + PV
        let tflops = flops / (ms as f64 * 1e-3) / 1e12;
        format!("{ms:.3} ms  {tflops:.1} TFLOP/s")
    } else {
        "".to_string()
    };

    let status = if l2_rel < 0.01 { "PASS" } else { "FAIL" };
    println!(
        "FLASH-ATTN [{status}] L={l_seq} S={s_seq} Dh={DH}  \
         L2-rel={l2_rel:.2e}  max_abs(|o|>0.1)={max_abs_big:.2e}  {timing}"
    );
    if l2_rel >= 0.01 {
        // Show a few mismatches for debugging:
        let mut shown = 0;
        for i in 0..l_seq * DH {
            if (o_ref[i] as f64 - o_gpu_f[i] as f64).abs() > 0.05 && shown < 4 {
                println!(
                    "  mismatch [{},{}] ref={:.4} gpu={:.4}",
                    i / DH, i % DH, o_ref[i], o_gpu_f[i]
                );
                shown += 1;
            }
        }
    }
    Ok(())
}

// ── Main ─────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    println!("=== Stage-0: f16 mma.sync spike ===");
    run_mma_f16_spike(&module, &stream)?;

    println!("\n=== Flash attention correctness ===");
    run_flash_attn(&module, &stream)?;

    Ok(())
}
