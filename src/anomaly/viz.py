"""Plots shared by all experiment scripts."""
import matplotlib

matplotlib.use("Agg")  # render to files only; no display window needed
import matplotlib.pyplot as plt

# Fixed colour per category, so a category looks the same in every figure.
# Colourblind-checked categorical palette, always assigned in this order.
CATEGORY_COLORS = {
    "carpet": "#2a78d6", "grid": "#eb6834", "leather": "#1baf7a",
    "bottle": "#eda100", "hazelnut": "#e87ba4", "screw": "#008300",
    "metal_nut": "#6250d6",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]  # second cue besides colour
TITLES = {"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC"}


def plot_metric_sweep(rows, param, metrics, path, log_x=False, xlabel=None):
    """One panel per metric, one line per category, `param` on the x-axis.

    rows: list of dicts, each with keys "category", param and every metric.
    """
    categories = list(dict.fromkeys(r["category"] for r in rows))  # keep order, drop repeats
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4),
                             sharey=True, squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for i, cat in enumerate(categories):
            pts = sorted((r[param], r[metric]) for r in rows if r["category"] == cat)
            xs, ys = zip(*pts)
            ax.plot(xs, ys, color=CATEGORY_COLORS.get(cat), marker=MARKERS[i % len(MARKERS)],
                    markersize=6, linewidth=2, label=cat)
        if log_x:
            ax.set_xscale("log")
            ticks = sorted({r[param] for r in rows})
            ax.set_xticks(ticks, [str(t) for t in ticks])  # label the swept values, not powers of 10
            ax.minorticks_off()
        ax.set_title(TITLES.get(metric, metric))
        ax.set_xlabel(xlabel or param)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("AUROC")
    axes[0, -1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
