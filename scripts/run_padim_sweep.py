"""PaDiM ablations: change one setting, keep everything else at the default.

    python scripts/run_padim_sweep.py --study layers   # layer1 / layer1+2 / layer1+2+3, d = 64 each
    python scripts/run_padim_sweep.py --study eps      # eps = 0.001, 0.01, 0.1, 1

Writes results/ablation_<study>.csv and results/figures/ablation_<study>.png.
"""
import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))   # reuse run_padim.run_category

from anomaly.viz import plot_metric_sweep  # noqa: E402
from run_padim import parse_args as padim_defaults, run_category  # noqa: E402

# each study: (setting overrides, extra columns for the CSV/plot)
STUDIES = {
    # d = 64 for all three, because layer1 alone has only 64 channels: equal dimension, fair comparison
    "layers": [({"layers": ["layer1"], "d": 64}, {"n_layers": 1}),
               ({"layers": ["layer1", "layer2"], "d": 64}, {"n_layers": 2}),
               ({"layers": ["layer1", "layer2", "layer3"], "d": 64}, {"n_layers": 3})],
    "eps": [({"eps": e}, {}) for e in (0.001, 0.01, 0.1, 1.0)],
}
PLOTS = {
    "layers": dict(param="n_layers", xlabel="layers used (d = 64 each)",
                   xticks={1: "layer1", 2: "layer1+2", 3: "layer1+2+3"}),
    "eps": dict(param="eps", xlabel="covariance regulariser eps", log_x=True,
                mark_x=0.01, mark_label="default"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--study", required=True, choices=STUDIES)
    p.add_argument("--categories", nargs="+", default=None, help="default: the five primary categories")
    p.add_argument("--plot-only", action="store_true", help="redraw the figure from the existing CSV")
    cli = p.parse_args()

    out = ROOT / "results" / f"ablation_{cli.study}.csv"
    rows = []
    param = PLOTS[cli.study]["param"]
    if cli.plot_only:
        with open(out, newline="") as f:           # CSV values are text: convert what the plot uses
            rows = [{**r, param: float(r[param]), "image_auroc": float(r["image_auroc"]),
                     "pixel_auroc": float(r["pixel_auroc"])} for r in csv.DictReader(f)]
    for overrides, extra in [] if cli.plot_only else STUDIES[cli.study]:
        args = padim_defaults([])                 # every run_padim default ...
        for key, value in overrides.items():      # ... except the setting under study
            setattr(args, key, value)
        for cat in cli.categories or args.categories:
            rows.append({**run_category(cat, args), **extra})
            print(rows[-1], flush=True)
            with open(out, "w", newline="") as f:  # rewrite after every row
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
    fig = ROOT / "results" / "figures" / f"ablation_{cli.study}.png"
    plot_metric_sweep(rows, metrics=["image_auroc", "pixel_auroc"], path=fig, **PLOTS[cli.study])
    print(f"wrote {out} and {fig}")


if __name__ == "__main__":
    main()
