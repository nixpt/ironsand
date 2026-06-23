//! CPU reference implementations for decode-role kernels
//!
//! Used for:
//! - Correctness validation (GPU vs CPU comparison)
//! - Testing without GPU hardware
//! - Baseline performance measurement
//! - Algorithm verification

/// RMS Normalization: CPU reference
///
/// Computes: output[i] = input[i] / sqrt(mean(input^2) + eps)
///
/// This is the standard RMS norm used in modern LLMs (LLaMA, GPT-style models).
/// It's numerically equivalent to layer normalization but without the learned
/// scale and shift parameters (those are applied separately).
///
/// # Arguments
/// * `input` - Input vector of any length
/// * `eps` - Small epsilon for numerical stability (prevents division by zero)
///
/// # Returns
/// Normalized output vector of same length as input
///
/// # Example
/// ```ignore
/// let input = vec![1.0, 2.0, 3.0];
/// let output = rms_norm_single(&input, 1e-6);
/// // RMS = sqrt((1 + 4 + 9) / 3 + eps) ≈ sqrt(4.67)
/// // output[0] ≈ 1.0 / sqrt(4.67) ≈ 0.463
/// ```
pub fn rms_norm_single(input: &[f32], eps: f32) -> Vec<f32> {
    let n = input.len();

    // Phase 1: Compute sum of squares
    let sum_sq: f32 = input.iter().map(|x| x * x).sum();

    // Phase 2: Compute RMS and its reciprocal
    let rms_sq = sum_sq / n as f32;
    let inv_rms = 1.0 / (rms_sq + eps).sqrt();

    // Phase 3: Normalize
    input.iter().map(|x| x * inv_rms).collect()
}

/// Thin GEMV (Matrix-Vector Product): CPU reference
///
/// Computes: output = matrix @ vector
/// where matrix is [m × n] and vector is [n], output is [m]
///
/// Used for:
/// - Projection layers (8192 → 8192 typical in Llama)
/// - Validation against GPU implementation
///
/// # Arguments
/// * `matrix` - Row-major matrix [m × n]
/// * `vector` - Input vector [n]
/// * `m` - Number of rows (output size)
/// * `n` - Number of columns (input size)
///
/// # Returns
/// Output vector of size m
pub fn gemv_decode_single(
    matrix: &[f32],
    vector: &[f32],
    m: usize,
    n: usize,
) -> Vec<f32> {
    assert_eq!(matrix.len(), m * n, "Matrix size mismatch");
    assert_eq!(vector.len(), n, "Vector size mismatch");

    let mut output = vec![0.0; m];

    for i in 0..m {
        let row = &matrix[i * n..(i + 1) * n];
        let dot_product: f32 = row.iter()
            .zip(vector.iter())
            .map(|(a, b)| a * b)
            .sum();
        output[i] = dot_product;
    }

    output
}

/// Flash Attention (Single Query): CPU reference
///
/// Computes attention for a single query over cached key/value.
/// Used in decode phase where we only compute attention for the latest token.
///
/// # Arguments
/// * `query` - Query vector [head_dim]
/// * `k_cache` - Cached keys [seq_len × head_dim]
/// * `v_cache` - Cached values [seq_len × head_dim]
/// * `head_dim` - Dimension of each head
/// * `seq_len` - Number of cached tokens
///
/// # Returns
/// Attention output [head_dim]
///
/// Formula:
/// ```
/// scores = query @ K_cache.T  // [seq_len]
/// attn_weights = softmax(scores)  // [seq_len]
/// output = attn_weights @ V_cache  // [head_dim]
/// ```
pub fn flash_attn_single(
    query: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    head_dim: usize,
    seq_len: usize,
) -> Vec<f32> {
    assert_eq!(query.len(), head_dim);
    assert_eq!(k_cache.len(), seq_len * head_dim);
    assert_eq!(v_cache.len(), seq_len * head_dim);

    // Phase 1: Compute attention scores (query @ K.T)
    let mut scores = vec![0.0; seq_len];
    for s in 0..seq_len {
        let key = &k_cache[s * head_dim..(s + 1) * head_dim];
        let score: f32 = query.iter()
            .zip(key.iter())
            .map(|(q, k)| q * k)
            .sum();
        scores[s] = score / (head_dim as f32).sqrt(); // Scale by sqrt(head_dim)
    }

    // Phase 2: Softmax (online algorithm to avoid overflow)
    let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum_exp = 0.0;
    for score in &mut scores {
        *score = (*score - max_score).exp();
        sum_exp += *score;
    }
    for score in &mut scores {
        *score /= sum_exp;
    }

    // Phase 3: Weighted sum of values (attn_weights @ V)
    let mut output = vec![0.0; head_dim];
    for s in 0..seq_len {
        let value = &v_cache[s * head_dim..(s + 1) * head_dim];
        for d in 0..head_dim {
            output[d] += scores[s] * value[d];
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_norm() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = rms_norm_single(&input, 1e-6);

        // Manual: rms_sq = (1+4+9+16)/4 = 30/4 = 7.5
        // inv_rms = 1 / sqrt(7.5) ≈ 0.365148
        let expected = vec![0.365148, 0.730297, 1.095445, 1.460594];

        for (o, e) in output.iter().zip(expected.iter()) {
            assert!((o - e).abs() < 1e-5, "Mismatch: {} vs {}", o, e);
        }
    }

    #[test]
    fn test_gemv_decode() {
        // Simple 2x2 matrix
        let matrix = vec![
            1.0, 2.0,  // row 0: [1, 2]
            3.0, 4.0,  // row 1: [3, 4]
        ];
        let vector = vec![1.0, 2.0];

        let output = gemv_decode_single(&matrix, &vector, 2, 2);

        // row0 @ vec = 1*1 + 2*2 = 5
        // row1 @ vec = 3*1 + 4*2 = 11
        assert!((output[0] - 5.0).abs() < 1e-6);
        assert!((output[1] - 11.0).abs() < 1e-6);
    }

    #[test]
    fn test_flash_attn_single() {
        let head_dim = 4;
        let seq_len = 3;

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let k_cache = vec![
            1.0, 0.0, 0.0, 0.0,  // key 0
            0.0, 1.0, 0.0, 0.0,  // key 1
            0.0, 0.0, 1.0, 0.0,  // key 2
        ];
        let v_cache = vec![
            1.0, 0.0, 0.0, 0.0,  // val 0
            0.0, 2.0, 0.0, 0.0,  // val 1
            0.0, 0.0, 3.0, 0.0,  // val 2
        ];

        let output = flash_attn_single(&query, &k_cache, &v_cache, head_dim, seq_len);

        // Query dot key 0: 1.0, key 1: 0.0, key 2: 0.0
        // After softmax: exp(1/2) ≈ 1.649, others ≈ 1.0
        // Weights should favor key 0
        // Output should be mostly [1.0, 0, 0, 0]

        assert!(output[0] > 0.5, "Attention should favor key 0");
    }
}
