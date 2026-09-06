"""Headless scientific plots; consume recorded measurements, never refit poses."""
from __future__ import annotations

import numpy as np


def comparison_plots(output, reports):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    variants = [name for name in ("droid_initial", "recam_candidate", "pointworld_release")
                if any(name in r["summary"] for r in reports)]
    colors = dict(droid_initial="#64748b", recam_candidate="#0c9d80", pointworld_release="#8555ca")
    names = dict(droid_initial="DROID initial", recam_candidate="ReCam candidate", pointworld_release="PointWorld release")
    metrics = [("depth", "point_weighted_m", 100., "Robot-depth L1 (cm) | lower is better"),
               ("two_view", "f1_5mm", 100., "Two-view F1 @ 5 mm (%) | higher is better"),
               ("two_view", "f1_20mm", 100., "Two-view F1 @ 20 mm (%) | higher is better")]
    if not reports:
        return
    if len(reports) <= 12:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
        x, width = np.arange(len(reports)), .75 / len(variants)
        for ax, (group, key, scale, title) in zip(axes, metrics):
            for j, name in enumerate(variants):
                yy = [r["summary"].get(name, {}).get(group, {}).get(key) for r in reports]
                xx = x + (j - (len(variants) - 1) / 2) * width
                valid = [q for q, v in enumerate(yy) if v is not None]
                bars = ax.bar(xx[valid], [yy[q] * scale for q in valid], width=width * .9,
                              label=names[name], color=colors[name])
                ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
            ax.set_title(title, fontsize=11)
            ax.set_xticks(x, [f"Episode {r['episode_index']}" for r in reports])
            ax.set_ylim(bottom=0)
            if group == "two_view":
                ax.set_ylim(0, 108)
            else:
                ax.margins(y=.18)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
        fig.legend(*axes[0].get_legend_handles_labels(), fontsize=9, loc="outside lower center", ncol=3)
        fig.suptitle("Same frames, depth and intrinsics | tiny sample: no population-level quality claim", fontsize=12)
        fig.savefig(output / "comparison.png", dpi=180)
        fig.savefig(output / "comparison.pdf")
        plt.close(fig)
    # Pair each candidate with the initial on exactly its own available cohort.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    for name in variants:
        if name == "droid_initial":
            continue
        for ax, (group, key, scale, title) in zip(axes, metrics):
            pairs = [r for r in reports if name in r["summary"] and
                     r["summary"][name][group][key] is not None and r["summary"]["droid_initial"][group][key] is not None]
            if not pairs:
                continue
            for variant, style, label in ((name, "-", names[name]), ("droid_initial", "--", f"Initial: same {names[name]} cohort")):
                yy = np.sort([r["summary"][variant][group][key] * scale for r in pairs])
                ax.step(yy, np.arange(1, len(yy) + 1) / len(yy), where="post", color=colors[name],
                        linestyle=style, label=f"{label} (n={len(yy)})")
            ax.set_xlabel(title)
            ax.set_ylabel("Fraction of episodes <= x")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle("Per-episode cumulative distributions | paired availability, no silent cohort mixing")
    fig.savefig(output / "paired_cdf.png", dpi=180)
    fig.savefig(output / "paired_cdf.pdf")
    plt.close(fig)
