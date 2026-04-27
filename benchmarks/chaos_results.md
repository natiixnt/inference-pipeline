# Chaos Engineering Results

Failure-injection runs against the v2.0 pipeline on the 8x H100 + 4x A100 + 4x L40S
cluster. The harness in `src/inference_pipeline/chaos.py` simulates the four failure
modes we see most often in production:

* GPU device failures (XID errors, NVLink fabric flaps treated as device drops)
* Network partitions (top-of-rack switch flaps)
* OOM events (CUDA OOM mid-request)
* Cascading failures (two GPUs within 100ms, fabric / PSU events)

Goal: failover under 2 seconds end-to-end, with no full process restart, and
request loss rate below 0.1% during the failover window.

## Single GPU Failure

500 concurrent in-flight requests when one GPU goes dark. Measured via the chaos
harness driving the live scheduler on a soak run.

| Metric | p50 | p95 | p99 |
|---|---:|---:|---:|
| Time to detection (heartbeat miss) | 280ms | 510ms | 720ms |
| Time to scheduler reaction | 10ms | 40ms | 70ms |
| Time to full reschedule (in-flight reassigned) | 1.18s | 1.62s | 1.84s |
| Throughput recovery (return to pre-failure tok/s) | 1.40s | 1.95s | 2.30s |
| **End-to-end recovery** | **1.40s** | **1.95s** | **2.30s** |
| Lost requests (returned 5xx) | 0 | 2 | 7 |
| Rescheduled requests (no client error) | 30 | 42 | 49 |

Recovery SLA met at p95 (1.95s vs 2.0s budget). The p99 misses by 300ms; the
extra time is dominated by KV cache warm-up on the surviving devices, not
scheduling. Loss rate 0.4% at p99.

## Network Partition

A subset of devices loses connectivity for 30 seconds, then comes back.

| Metric | Value |
|---|---:|
| Devices partitioned | 2 (1x A100, 1x L40S) |
| In-flight requests at partition time | ~140 |
| Rescheduled to peer devices | 137 |
| Lost (timed out before reschedule) | 3 |
| End-to-end recovery (after reattach) | 1.6s |
| KV cache cold reuse on reattached devices | 91% (radix prefix cache) |

The radix prefix cache earned its keep here: the partitioned-then-reattached
devices recovered 91% of their pre-partition prefix-cache hits without recomputing
prefill. Without the radix cache, p95 TTFT after reattach jumps from 71ms to 184ms
for the first 30 seconds.

## OOM Recovery

Triggered by oversubscribing the KV pool on H100-3 to 105% of capacity.

| Metric | Value |
|---|---:|
| Time to OOM detection | 8ms (CUDA error path) |
| Lowest-priority sequences evicted | 4 |
| KV memory freed | 7.2 GB |
| Time to resume forward passes | 340ms |
| Lost requests | 0 (all eviction victims retried successfully) |

Importantly: no process restart. The KV evictor freed 7.2 GB of low-priority
sequence state, the scheduler retried those sequences from their last completed
token, and forward passes resumed. The retried requests saw an extra ~340ms of
TTFT but completed successfully.

## Cascading Failure

Worst case we have hardware to test: two GPUs fail within 100ms.

| Metric | Value |
|---|---:|
| GPUs failed | H100-2, H100-3 (cascade scenario) |
| In-flight requests across both | 95 |
| Rescheduled successfully | 92 |
| Lost requests | 3 |
| Loss rate | 3.16% |
| Recovery time | 2.4s |

This one busts the 2.0s SLA (2.4s p95). The slowdown is the survivor-side
memory pressure: with 6 H100s instead of 8 carrying the same load, the KV pool
runs hot and the scheduler has to evict more low-priority sequences before it
can place the rescheduled work. We could fix this by reserving 12% headroom on
each device specifically for cascade absorption, but that costs steady-state
throughput. Open trade-off; not committed yet.

## Soak Run

24 hours, chaos enabled at the default 5% per-tick failure probability. Exposed
to the production traffic shadow (read-only mirror of production requests).

| Metric | Value |
|---|---:|
| Failures injected | 4,182 |
| GPU failures | 1,659 |
| OOM events | 1,256 |
| Network partitions | 622 |
| NCCL hangs | 421 |
| Cascading failures | 224 |
| Recoveries that met SLA | 4,012 (96.0%) |
| SLA violations (recovery > 2.0s) | 170 (4.0%) |
| Total request loss rate over the soak | 0.043% |
| Process restarts triggered | 0 |

The 4% SLA violation rate is dominated by cascading failures (54% of violations)
and by recoveries during the rare moments when all four L40S nodes were already
unhealthy. None of the runs required a process restart.

## Reproducing

```bash
# Quick run (~30 seconds, in-process, no real GPU needed)
python -m inference_pipeline.chaos --quick

# Full soak (24 hours, requires a live cluster)
python -m benchmarks.runner \
  --chaos-enabled \
  --chaos-failure-probability 0.05 \
  --chaos-allow-cascading \
  --duration 86400 \
  --traffic-source production-shadow
```

Raw events are in `benchmarks/chaos_events.jsonl` after a run (rolled out of git
because of size; ~150 MB per soak).

## What This Did Not Catch

Honest gap list, since chaos engineering is only as good as its coverage:

* **Silent data corruption.** Bit flips in HBM that don't trigger ECC. We don't
  have a checksum on KV cache contents, so a flipped bit produces a wrong token
  rather than a detected fault.
* **Driver bugs.** A real CUDA driver bug looks nothing like our injected
  symptoms. The harness can only model what we already know how to recognize.
* **Power events.** Whole-rack power loss is simulated by partitioning all
  GPUs at once, but in practice the failover involves another rack picking up
  the load, which we don't model here.

We add new injection modes whenever we hit a novel production failure that the
existing modes wouldn't have caught.
