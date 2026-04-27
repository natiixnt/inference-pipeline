"""Generate benchmark visualization charts.

Produces four PNG figures used in the README and benchmarks documentation. Numbers
match the tables in benchmarks/README.md (collected on the 8x H100 + 4x A100 + 4x L40S
cluster), so changes here should track changes there.

Run with:

    python benchmarks/generate_charts.py

Output goes to benchmarks/charts/.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# matplotlib style choices: small figures (no Retina blur on GitHub), tight layout,
# and tableau colors that survive both light and dark README rendering.
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 130,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
    }
)

CHARTS_DIR = Path(__file__).parent / "charts"


def _ensure_outdir() -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHARTS_DIR


def throughput_comparison(outdir: Path) -> Path:
    """Bar chart: tokens/sec at 500 concurrency vs competitors.

    Uses the @500 conc column from benchmarks/README.md throughput table. We
    highlight our system (orange) vs everything else (gray) to make the gap
    pop in the README thumbnail.
    """
    systems = [
        "Naive\n(no batching)",
        "Triton +\nFasterTransformer",
        "TGI 2.0",
        "vLLM 0.6",
        "SGLang 0.3",
        "TensorRT-LLM\n0.10",
        "This system",
    ]
    # tokens/sec at 500 concurrent. SGLang and TRT-LLM numbers added from
    # benchmarks/sota_comparison.md once that landed.
    tps = [5_800, 78_200, 131_800, 142_600, 198_400, 232_700, 244_800]

    colors = ["#9aa0a6"] * (len(systems) - 1) + ["#ff7043"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(systems, tps, color=colors, edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, tps):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 3000,
            f"{value/1000:.1f}k",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput at 500 Concurrent Sessions (Llama-3-70B)")
    ax.set_ylim(0, max(tps) * 1.15)
    ax.tick_params(axis="x", labelsize=8.5)

    out = outdir / "throughput_comparison.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def ttft_distribution(outdir: Path) -> Path:
    """Line chart: TTFT p50/p95/p99 across concurrency levels.

    Pulled from the TTFT table in benchmarks/README.md, with the 1000 conc
    point extrapolated from the soak test. The whole point of this chart is
    to show TTFT scales sublinearly with concurrency thanks to the radix
    prefix cache (see benchmarks/README.md for the explanation).
    """
    concurrency = np.array([100, 250, 500, 1000])
    p50 = np.array([12, 19, 29, 47])
    p95 = np.array([31, 48, 62, 108])
    p99 = np.array([52, 71, 104, 188])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(concurrency, p50, marker="o", linewidth=2.2, color="#1f77b4", label="p50")
    ax.plot(concurrency, p95, marker="s", linewidth=2.2, color="#ff7f0e", label="p95")
    ax.plot(concurrency, p99, marker="^", linewidth=2.2, color="#d62728", label="p99")

    # SLA reference line: 100ms is the standard interactive-chat target
    ax.axhline(100, color="#2ca02c", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Interactive SLA (100ms)")

    for x, y in zip(concurrency, p95):
        ax.annotate(f"{y}ms", (x, y), textcoords="offset points",
                    xytext=(6, 6), fontsize=8.5, color="#ff7f0e")

    ax.set_xlabel("Concurrent Sessions")
    ax.set_ylabel("Time To First Token (ms)")
    ax.set_title("TTFT Distribution Under Load")
    ax.set_xscale("log")
    ax.set_xticks(concurrency)
    ax.set_xticklabels([str(c) for c in concurrency])
    ax.legend(loc="upper left", framealpha=0.9)

    out = outdir / "ttft_distribution.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def gpu_utilization(outdir: Path) -> Path:
    """Heatmap: SM utilization across H100 / A100 / L40S devices.

    Same numbers as the table in benchmarks/README.md GPU Utilization section.
    The heatmap layout (rows = device tier, cols = device index) makes the
    H100/A100/L40S tiering immediately visible.
    """
    devices = ["GPU-0", "GPU-1", "GPU-2", "GPU-3", "GPU-4", "GPU-5", "GPU-6", "GPU-7"]
    tiers = ["H100", "A100", "L40S"]

    # NaN means "no device of that tier at that index" - shown as a blank cell
    util = np.array(
        [
            [93, 92, 91, 91, 92, 91, 90, 89],
            [86, 85, 84, 83, np.nan, np.nan, np.nan, np.nan],
            [79, 78, 77, 76, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    fig, ax = plt.subplots(figsize=(9, 3.6))
    cmap = plt.get_cmap("RdYlGn")
    cmap.set_bad(color="#eeeeee")

    masked = np.ma.masked_invalid(util)
    im = ax.imshow(masked, cmap=cmap, vmin=60, vmax=100, aspect="auto")

    ax.set_xticks(np.arange(len(devices)))
    ax.set_xticklabels(devices, fontsize=9)
    ax.set_yticks(np.arange(len(tiers)))
    ax.set_yticklabels(tiers, fontsize=10, fontweight="bold")
    ax.set_title("SM Utilization Heatmap @ 500 Concurrent (DCGM, 100ms granularity)")

    # Annotate cells with the actual percent (skip masked cells)
    for i in range(util.shape[0]):
        for j in range(util.shape[1]):
            if np.isnan(util[i, j]):
                continue
            color = "white" if util[i, j] > 88 else "black"
            ax.text(j, i, f"{int(util[i, j])}%", ha="center", va="center",
                    color=color, fontsize=9, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("SM Utilization (%)", fontsize=9)

    ax.set_xticks(np.arange(len(devices) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(tiers) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    out = outdir / "gpu_utilization.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def cost_per_million(outdir: Path) -> Path:
    """Bar chart: $/1M tokens vs competitors at scale.

    Cost = (cluster $/hr) / (tokens/hr per million). At 500 conc using on-demand
    H100 pricing ($3.50/hr/GPU) on a 16 GPU cluster, our $0.05 number comes
    straight out of benchmarks/README.md "Cost per 1M Tokens".
    """
    systems = ["Naive", "Triton + FT", "TGI 2.0", "vLLM 0.6", "SGLang 0.3",
               "TensorRT-LLM", "This system"]
    cost = [2.40, 0.72, 0.11, 0.10, 0.07, 0.06, 0.05]

    colors = ["#9aa0a6"] * (len(systems) - 1) + ["#ff7043"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(systems, cost, color=colors, edgecolor="black", linewidth=0.6)

    for bar, value in zip(bars, cost):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.04,
            f"${value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylabel("USD per 1M tokens")
    ax.set_title("Cost Efficiency at 500 Concurrent (lower is better)")
    ax.set_ylim(0, max(cost) * 1.15)
    ax.tick_params(axis="x", labelsize=9)

    out = outdir / "cost_per_million.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    outdir = _ensure_outdir()
    paths = [
        throughput_comparison(outdir),
        ttft_distribution(outdir),
        gpu_utilization(outdir),
        cost_per_million(outdir),
    ]
    for p in paths:
        size_kb = os.path.getsize(p) / 1024
        print(f"wrote {p.relative_to(Path(__file__).parent.parent)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    main()
