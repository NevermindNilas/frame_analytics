"""Plot the saved CPU optimization confirmation; no benchmarks are rerun."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


def label(metric, backward):
    names = {"ssim": "SSIM", "ms_ssim": "MS-SSIM", "ssimulacra2": "SSIMULACRA 2"}
    return names[metric] + (" · training" if backward else " · forward")


def panel(ax, labels, before, after, title, unit):
    y = np.arange(len(labels))
    ax.barh(y - 0.18, before, height=0.31, color="#64748b")
    ax.barh(y + 0.18, after, height=0.31, color="#008577")
    xmax = max(before) * 1.27
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(b + xmax * 0.015, i - 0.18, f"{b:,.1f}", va="center", fontsize=10)
        ax.text(a + xmax * 0.015, i + 0.18, f"{a:,.1f}", va="center", fontsize=10,
                color="#00685e", fontweight="bold")
    ax.set(yticks=y, yticklabels=labels, xlim=(0, xmax), xlabel=unit)
    ax.invert_yaxis()
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", pad=15)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e4e8ed", linewidth=0.8)
    ax.tick_params(axis="both", length=0, labelsize=10, pad=8)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results" / "cpu_filters_2026-09-12")
    parser.add_argument("--output", type=Path, default=Path("bench/figures"))
    args = parser.parse_args()
    results = json.loads((args.results / "confirmation.json").read_text())
    memory = json.loads((args.results / "memory.json").read_text())
    plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": "#172331",
                         "axes.labelcolor": "#344253", "xtick.color": "#344253",
                         "ytick.color": "#172331", "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.6))
    fig.subplots_adjust(left=0.155, right=0.975, top=0.81, bottom=0.13,
                        wspace=0.59, hspace=0.64)
    fig.suptitle("CPU optimization: before vs. after", x=0.035, y=0.974,
                 ha="left", fontsize=25, fontweight="bold")
    fig.text(0.035, 0.921,
             "Portable path  ·  float32 RGB  ·  i7-13700K, 16 threads  ·  compilation disabled",
             fontsize=13, color="#475569")
    fig.legend(handles=[Patch(color="#64748b", label="Before"),
                        Patch(color="#008577", label="After")],
               loc="upper left", bbox_to_anchor=(0.03, 0.894), ncol=2,
               frameon=False, fontsize=12)
    fig.text(0.975, 0.868, "Lower is better", ha="right", fontsize=12,
             fontweight="bold")
    shapes = [(1, 3, 512, 512), (1, 3, 720, 1280), (2, 3, 257, 389)]
    titles = ["Latency · 512 × 512, batch 1", "Latency · 720p, batch 1",
              "Latency · 257 × 389, batch 2"]
    for ax, shape, title in zip(axes.flat, shapes, titles):
        cases = [c for c in results["cases"] if tuple(c["shape"]) == shape]
        assert len(cases) == 5
        panel(ax, [label(c["metric"], c["backward"]) for c in cases],
              [c["baseline_ms"] for c in cases], [c["candidate_ms"] for c in cases],
              title, "Milliseconds per call")
    pairs = list(zip(memory[::2], memory[1::2]))
    assert all(b["arm"] == "baseline" and a["arm"] == "candidate" and
               b["metric"] == a["metric"] and b["backward"] == a["backward"]
               for b, a in pairs)
    panel(axes[1, 1], [label(b["metric"], b["backward"]) for b, _ in pairs],
          [b["delta_mib"] for b, _ in pairs], [a["delta_mib"] for _, a in pairs],
          "Extra peak process memory · 720p, batch 1", "MiB above the input baseline")
    fig.text(0.035, 0.053,
             "Latency: median of 15 paired samples, 3 calls each. Training includes gradients for both inputs.",
             fontsize=10, color="#475569")
    fig.text(0.035, 0.029,
             "Memory: fresh-process high-water growth across the first 3 calls, including lazy setup. Panel axes differ.",
             fontsize=10, color="#475569")
    args.output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        path = args.output / f"cpu_filters_before_after.{ext}"
        fig.savefig(path, dpi=150, facecolor="white")
        if ext == "svg":
            path.write_text("\n".join(line.rstrip() for line in
                                      path.read_text(encoding="utf-8").splitlines()) + "\n",
                            encoding="utf-8")
        print(path.resolve())
    plt.close(fig)


if __name__ == "__main__":
    main()
