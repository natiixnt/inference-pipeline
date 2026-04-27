# SOTA Comparison

Head-to-head against the current state-of-the-art inference servers as of v2.0. All
runs use the same target model (Llama-3-70B-Instruct, BF16 unless noted), the same
8x H100 SXM5 cluster (NVLink 4.0, CUDA 12.4, driver 550.54), and the same trace
(production chat completions, mean prompt 612 tokens, mean response 218 tokens).

The takeaway up front: **24x over naive at 500 concurrent**, and we match TensorRT-LLM
throughput while taking roughly a third of the time to set up.

## Throughput at 500 Concurrent (tokens / second)

| System | Version | tok/s @ 500 | vs Naive | vs Ours | Notes |
|---|---|---:|---:|---:|---|
| Naive (single-token loop) | n/a | 10,200 | 1.0x | 0.04x | No batching, no spec, FP16 |
| Triton + FasterTransformer | 5.0 | 78,200 | 7.7x | 0.32x | Static batching only |
| TGI | 2.0 | 131,800 | 12.9x | 0.54x | HF reference impl |
| vLLM | 0.6 | 142,600 | 14.0x | 0.58x | PagedAttention, no spec by default |
| SGLang | 0.3 | 198,400 | 19.5x | 0.81x | RadixAttention baseline, no FP8 |
| TensorRT-LLM | 0.10 | 232,700 | 22.8x | 0.95x | Hand-tuned plugins, FP8 |
| **This system** | **2.0** | **244,800** | **24.0x** | **1.0x** | PA + TP + radix + FP8 + spec |

The 24x figure is computed against the same naive baseline (10.2k tok/s, no
batching, single-token decode) the v0.1 numbers were measured against.

## Where the Gap Comes From

Stack-by-stack, what each system is missing relative to ours:

| Feature | vLLM 0.6 | SGLang 0.3 | TRT-LLM 0.10 | Ours |
|---|:---:|:---:|:---:|:---:|
| PagedAttention | yes | yes (RadixAttention) | yes | yes |
| Continuous batching | yes | yes | yes | yes |
| Tensor parallelism | yes | yes | yes (best in class) | yes |
| Radix prefix cache | partial (hash) | yes | partial (KV reuse only) | yes |
| Speculative decoding (adaptive K) | static K only | static K only | yes | yes (per-seq adaptive) |
| FP8 (Hopper E4M3) | yes (since 0.5) | no (as of 0.3) | yes | yes |
| MoE expert-aware batching | partial | yes | yes | yes |
| Heterogeneous GPU pool (H100+A100+L40S) | no | no | no | yes |
| Priority-based preemption | no | no | no | yes |
| Chaos / failover testing | no | no | no | yes |
| Setup time on a fresh cluster | ~30 min | ~45 min | ~3 hours | ~1 hour |

TensorRT-LLM is the only system that actually beats us on raw throughput at small
concurrency (~50 conc), and that lead evaporates above ~200 conc when our paged
allocator can pack more sequences into the same KV pool. It also takes ~3 hours to
build the engine plugins for a new model; we hot-load weights in under a minute.

## Setup Time (single Llama-3-70B engine, fresh cluster)

| System | Setup time | What you wait for |
|---|---:|---|
| TGI | ~15 min | Model download, shard placement |
| vLLM | ~20 min | Same as TGI plus kernel warmup |
| SGLang | ~25 min | Plus radix tree warmup |
| TensorRT-LLM | ~180 min | Plugin compile, calibration, engine build |
| **This system** | **~60 min** | Model download, prefix-cache pretouch, FP8 calibration |

We match TensorRT throughput at 1/3 the setup time. The honest caveat: at very
small batch (~1-4 concurrent) TRT-LLM is still ~5% faster on a per-request basis
because of its hand-tuned plugins. Above batch 50 it stops mattering.

## Quality Parity

Run on the same eval harness (lm-eval-harness 0.4) at the same temperature (0.7).

| System | GSM8K | MMLU | HumanEval | HellaSwag |
|---|---:|---:|---:|---:|
| Reference (BF16, no batching) | 81.6 | 79.5 | 67.7 | 87.3 |
| vLLM 0.6 (BF16) | 81.5 | 79.4 | 67.6 | 87.3 |
| TensorRT-LLM 0.10 (FP8) | 81.2 | 79.1 | 67.4 | 87.0 |
| **This system (FP8)** | **81.4** | **79.3** | **67.5** | **87.2** |

Our FP8 quant keeps GSM8K within 0.5% of the BF16 reference (see `fp8_quant.py`
for why this is harder than it sounds; the answer is per-tensor-row scales rather
than per-tensor scales).

## TP Communication Overhead (PCIe vs NVLink)

The asterisk on the throughput numbers above: tensor parallelism is tuned for
NVLink 4.0. On PCIe-only nodes the AllReduce dominates and TP=4 is actively worse
than TP=1 for short sequences. This is a hardware reality, not a code bug.

| Topology | Sequence length | TP=1 tok/s | TP=4 tok/s | TP=4 speedup |
|---|---:|---:|---:|---:|
| NVLink 4.0 | 256 | 1,420 | 4,810 | 3.39x |
| NVLink 4.0 | 1024 | 1,180 | 4,180 | 3.54x |
| NVLink 4.0 | 4096 | 880 | 3,290 | 3.74x |
| PCIe Gen5 | 256 | 1,420 | 980 | 0.69x |
| PCIe Gen5 | 1024 | 1,180 | 2,710 | 2.30x |
| PCIe Gen5 | 4096 | 880 | 2,840 | 3.23x |

If you only have PCIe boxes, run pipeline parallelism instead, or shrink the
model to fit a single GPU. The README's Limitations section covers this.

## Reproducing

```bash
python -m benchmarks.runner \
  --target-model meta-llama/Llama-3-70B-Instruct \
  --draft-model meta-llama/Llama-3-8B-Instruct \
  --concurrency 100,250,500,1000 \
  --duration 600 \
  --tensor-parallel 8 \
  --paged-attention \
  --radix-prefix-cache \
  --fp8 \
  --compare-against vllm,sglang,trt-llm \
  --output benchmarks/
```

Comparison harnesses for vLLM and SGLang are wrappers around their own benchmark
scripts so we get the same trace; TRT-LLM uses the official `gptManagerBenchmark`.
