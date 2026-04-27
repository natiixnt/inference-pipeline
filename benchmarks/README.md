# Performance Benchmarks

All numbers collected on a mixed cluster: 8x H100 SXM5 (80GB), 4x A100 (80GB), 4x L40S (48GB).
Load generation via custom async harness (not locust, that thing can't keep up past 200 concurrent).
Target model: Llama-3-70B-Instruct, Draft model: Llama-3-8B-Instruct.

## v2.0 Improvements (PagedAttention + Tensor Parallel + Radix Prefix Cache)

Before/after comparison showing the impact of v2.0 optimizations:

| Metric | v1.0 | v2.0 | Improvement |
|--------|------|------|-------------|
| Throughput vs naive (500 conc) | 8.1x | 12.3x | +52% |
| Throughput vs vLLM (500 conc) | 1.39x | 1.8x | +29% |
| TTFT p95 @ 500 concurrent | 91ms | 62ms | -32% |
| GPU SM utilization (avg) | 85% | 91% | +6pp |
| KV-cache memory reduction | 41% | 58% | +17pp |
| Cost per 1M tokens (500 conc) | $0.08 | $0.05 | -37% |
| Speculative acceptance (8B/70B) | 0.82 | 0.87 | +6% |

**What changed:**
- PagedAttention eliminates internal fragmentation in KV-cache. No more pre-allocating max_seq_len blocks per request. Memory savings compound with concurrent requests because freed blocks are immediately reusable.
- Tensor parallelism across 8x H100 with NVLink 4.0 gives us near-linear scaling. Column-parallel on QKV, row-parallel on output. AllReduce overhead under 4% on NVLink.
- Radix prefix cache replaces the flat hash table. O(prefix_len) lookup finds the longest matching cached prefix, enabling partial prefix reuse that the old approach missed entirely.
- Adaptive draft length in speculative decoding now uses per-token acceptance history to adjust K dynamically, pushing acceptance from 82% to 87%.

## Throughput Comparison

| System | tok/s @ 100 conc | tok/s @ 250 conc | tok/s @ 500 conc | Peak tok/s |
|--------|-------------------|-------------------|-------------------|------------|
| **This system** | 58,900 | 138,200 | 244,800 | 262,000 |
| vLLM 0.4.1 | 41,800 | 89,200 | 142,600 | 158,000 |
| TGI 2.0 | 38,400 | 82,100 | 131,800 | 146,000 |
| Triton + FasterTransformer | 24,600 | 51,300 | 78,200 | 86,000 |
| Naive (no batching) | 5,800 | 5,800 | 5,800 | 5,800 |

Our system hits 12.3x over naive at 500 concurrent and 1.8x over vLLM. PagedAttention lets us
pack more sequences into GPU memory simultaneously (no fragmentation waste), and tensor
parallelism with NVLink AllReduce means we actually use all 8 H100s efficiently instead of
leaving compute on the table. The speculative decoding loop fills in the gaps - verify 5-7
tokens per target forward and immediately pack new requests into freed page table slots.

## TTFT Distribution (Time-to-First-Token)

Measured under sustained load for 10 minutes per concurrency level.

| Concurrency | p50 (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|-------------|----------|----------|----------|----------|----------|
| 100 | 12 | 24 | 31 | 52 | 98 |
| 250 | 19 | 38 | 48 | 71 | 142 |
| 500 | 29 | 51 | 62 | 104 | 228 |

Sub-65ms TTFT at p95 even at 500 concurrent. The radix prefix cache is the hero here - it
finds the longest matching cached prefix in O(prefix_len) time, so we skip prefill for not
just exact prefix matches but also partial overlaps. A request sharing 80% of a cached prefix
still benefits from that 80%. Without the radix cache, p95 at 500 concurrent jumps to ~195ms.

## Speculative Decoding Acceptance Rates

Acceptance rate = fraction of draft tokens accepted by target model verification.
Higher is better (means draft model predicts target distribution well).

| Draft Model | Target Model | Acceptance Rate | Effective Speedup | Optimal K |
|-------------|-------------|-----------------|-------------------|-----------|
| Llama-3-8B | Llama-3-70B | 0.87 | 3.1x | 6 |
| Llama-3-8B | Llama-3-70B (low temp) | 0.94 | 3.5x | 8 |
| Llama-3-1B | Llama-3-70B | 0.69 | 2.1x | 4 |
| Phi-3-mini | Llama-3-70B | 0.63 | 1.9x | 3 |
| Gemma-2B | Llama-3-70B | 0.66 | 2.0x | 4 |
| Llama-3-8B | Mixtral-8x22B | 0.79 | 2.7x | 5 |

The 8B/70B pair acceptance jumped from 82% to 87% with adaptive draft length. Instead of
fixed K=5, we now track per-sequence acceptance history and extend K when the draft model is
nailing it (up to K=8 for easy continuations) or shorten it when acceptance craters (K=3 for
hard reasoning). The effective speedup of 3.1x means every target forward pass produces 3.1
tokens on average.

## GPU Utilization Heatmap

Under sustained 500 concurrent sessions, measured via DCGM at 100ms granularity.

```
Device         | SM Util % | Mem BW Util % | Mem Allocated % | Tensor Core %
H100-0 (lead)  |    93     |      88       |       94        |      90
H100-1         |    92     |      87       |       93        |      89
H100-2         |    91     |      86       |       92        |      88
H100-3         |    91     |      85       |       91        |      88
H100-4         |    92     |      87       |       93        |      89
H100-5         |    91     |      86       |       92        |      88
H100-6         |    90     |      85       |       91        |      87
H100-7         |    89     |      84       |       90        |      86
A100-0         |    86     |      80       |       89        |      83
A100-1         |    85     |      79       |       88        |      82
A100-2         |    84     |      78       |       87        |      81
A100-3         |    83     |      77       |       86        |      80
L40S-0         |    79     |      74       |       83        |      76
L40S-1         |    78     |      73       |       82        |      75
L40S-2         |    77     |      72       |       81        |      74
L40S-3         |    76     |      71       |       80        |      73
```

Average cluster utilization: 91% SM on H100s, 85% on A100s, 78% on L40S. PagedAttention
eliminates the memory fragmentation that used to leave 15-20% of KV-cache capacity stranded
as unusable fragments. Now every freed block is immediately available, which means the
scheduler can pack more sequences and keep the SMs busier. Tensor parallelism ensures all
8 H100s share the load evenly with <4% communication overhead via NVLink 4.0.

## KV-Cache Sharing Savings

Prefix deduplication measured on production traffic (system prompts + few-shot examples).
v2.0 uses RadixTree prefix matching for partial prefix reuse.

| Metric | Value |
|--------|-------|
| Unique prefix clusters | 1,247 |
| Average prefix length | 1,412 tokens |
| Requests sharing a prefix | 81% |
| Memory saved per shared request | 462 MB (70B model) |
| Total memory freed by dedup | 52.8 GB across cluster |
| Radix cache hit rate | 0.94 |
| Effective KV memory reduction | 58% |

PagedAttention + radix prefix cache is the combo that gets us from 41% to 58% memory
reduction. The paged allocator eliminates internal fragmentation (no more wasted tail blocks),
and the radix cache finds partial prefix matches that the old hash table missed. A request
that shares 900 of 1200 prefix tokens with a cached entry now reuses those 900 tokens of
KV-cache instead of recomputing everything because the hash didn't match exactly.

## Cost per 1M Tokens

Based on on-demand H100 pricing ($3.50/hr/GPU) and measured throughput.

| Concurrency | Tokens/hr (cluster) | Cost / 1M tokens | vs vLLM | vs TGI |
|-------------|--------------------:|------------------:|--------:|-------:|
| 100 | 212.0M | $0.26 | -38% | -46% |
| 250 | 497.5M | $0.11 | -44% | -52% |
| 500 | 881.3M | $0.05 | -48% | -56% |

At 500 concurrent we're at $0.05/1M tokens. The 37% cost reduction from v1.0 comes from three
compounding effects: (1) paged attention packs more sequences per GPU, (2) radix prefix cache
eliminates redundant prefill compute, (3) adaptive speculative decoding produces more tokens
per forward pass. The autoscaler keeps utilization at 91% instead of the old 85%.

## Reproducing

```bash
# spin up the benchmark harness (needs GPU access)
python -m benchmarks.runner \
  --target-model meta-llama/Llama-3-70B-Instruct \
  --draft-model meta-llama/Llama-3-8B-Instruct \
  --concurrency 100,250,500 \
  --duration 600 \
  --tensor-parallel 8 \
  --paged-attention \
  --radix-prefix-cache \
  --output benchmarks/
```

Raw data in the JSON files in this directory. Visualization notebooks in `notebooks/` (WIP).
