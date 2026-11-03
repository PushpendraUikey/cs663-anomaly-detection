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


def plot_metric_sweep(rows, param, metrics, path, log_x=False, xlabel=None,
                      mark_x=None, mark_label=None):
    """One panel per metric, one line per category, `param` on the x-axis.

    rows: list of dicts, each with keys "category", param and every metric.
    mark_x: optional x value to highlight with a dashed line (e.g. the chosen k).
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
        if mark_x is not None:
            ax.axvline(mark_x, color="#888888", linestyle="--", linewidth=1, zorder=0)
            if mark_label:
                ax.annotate(mark_label, (mark_x, 0), xycoords=("data", "axes fraction"),
                            xytext=(-4, 4), textcoords="offset points", ha="right",
                            color="#555555", fontsize=9)
        ax.set_title(TITLES.get(metric, metric))
        ax.set_xlabel(xlabel or param)
        ax.grid(True, color="#e5e5e5", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("AUROC")
    # legend outside the plot area so it can never cover a line
    axes[0, -1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
