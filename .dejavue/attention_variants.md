# Attention Variants Arc

## GQA v1 (Grouped-Query Attention) — COMPLETE ✓

**Status**: Correctness verified. Ready for performance analysis.

**What it is**: Multi-head attention with K/V head sharing.
- Q: H query heads (e.g., H=12)
- K/V: G KV heads (e.g., G=4), where G < H
- Mapping: each query head group of size H/G shares the same KV head
- Use case: Llama 2, GPT-3.5+ (reduces KV cache by H/G, improves memory bandwidth)

**Implementation**:
- Based on v7 FlashAttention (vectorized Q/K loads, V swizzle, scalar PV)
- Grid: (query_tiles, query_heads) — more blocks, but less KV per block
- KV head mapping: `kv_head = query_head * num_kv_heads / num_query_heads`
- No masking or complex logic needed; pure tensor reshaping

**Correctness**: ✓ PASS
- Test: H=12 query heads, G=4 KV heads (3:1 ratio), L=64, S=64
- Relative error: 7.52e-5 (f16 quantization noise)

**Performance**: Not yet characterized
- Expected: Reduced memory traffic (K/V loads are 1/3 less than standard)
- Estimated speedup: ~10-20% for KV-bound workloads
- TBD: Measure vs standard attention

## Next Variants

**MQA (Multi-Query Attention)** — Extreme case of GQA
- G=1: All H query heads share the single KV head
- Even larger memory savings (H× reduction in KV cache)
- Simple kernel modification: GQA with G=1

**Sparse attention** — *Not pursued yet*
- Local (sliding window): Only attend to [i - W, i + W]
- Block-sparse: Predefined sparsity patterns
- Challenge: Per-element masking in MMA tile layout is complex
- Would require careful index-to-thread mapping or smart KV loading
- Deferred: Complexity/benefit tradeoff unclear for current tile shapes

## Lessons Learned

1. **GQA is straightforward** — pure tensor shape change, no algorithm modifications
2. **Sparse attention is intricate** — Per-element masking in FlashAttention's MMA layout requires care with thread-to-index mapping
3. **Head mapping is cheap** — The `query_head → kv_head` computation is a single integer division, negligible overhead

## Measurements TODO

- [ ] GQA H=12 G=4 vs standard H=12 (same total heads, reduced KV)
- [ ] GQA H=32 G=8 vs standard H=32
- [ ] MQA H=32 G=1 vs GQA
- [ ] Profile: is improvement purely from reduced memory, or does cache line efficiency matter?
