"""011.2 — Leave-one-day-out.

Carrega os JSONs do 011 (pixel e phase), e pra cada dia (em cada batch)
remove esse dia e recalcula MK p-value. Mostra qual dia carrega a tendência
sistêmica em april (ou em february).

Saída:
- CSV com (frontend, feature, batch, day_removed, mk_p, sp_p, lin_p).
- Plot: pra cada feature de interesse, barra horizontal mostrando p quando
  remove cada dia. Linha vertical em α=0.05.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.modal.trends import all_trend_tests


PIXEL_JSON = "out/experimentos/011/results_pixel.json"
PHASE_JSON = "out/experimentos/011/results_phase.json"
OUT_DIR = "out/experimentos/011-2/"

FEATURES_OF_INTEREST = [
    ("pixel", "energy_0.5_2.0_mean"),
    ("pixel", "energy_0.5_2.0_median"),
    ("pixel", "centroid_0.5_2.0_median"),
    ("phase", "energy_2.0_5.0_mean"),
    ("phase", "energy_0.5_2.0_median"),
    ("phase", "frac_ratio_low_over_high_median"),
]


def load_series(json_path):
    with open(json_path) as f:
        return json.load(f)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    sources = {
        "pixel": load_series(PIXEL_JSON),
        "phase": load_series(PHASE_JSON),
    }

    rows = []
    for frontend, feature in FEATURES_OF_INTEREST:
        data = sources[frontend]
        s = data["series"][feature]
        for batch in ("february", "april"):
            x_all = np.asarray(s[batch]["x"], dtype=float)
            y_all = np.asarray(s[batch]["y"], dtype=float)
            days_unique = sorted(set(x_all.tolist()))
            for d in days_unique:
                keep = x_all != d
                if keep.sum() < 4:
                    continue
                tests = all_trend_tests(y_all[keep].tolist(), x=x_all[keep].tolist())
                rows.append({
                    "frontend": frontend,
                    "feature": feature,
                    "batch": batch,
                    "day_removed": int(d),
                    "n_remaining": int(keep.sum()),
                    "mk_p": tests["mann_kendall"].p_value,
                    "mk_tau": tests["mann_kendall"].statistic,
                    "sp_p": tests["spearman"].p_value,
                    "lin_p": tests["linear"].p_value,
                })

    csv_path = os.path.join(OUT_DIR, "leave_one_day_out.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[CSV] {csv_path}")

    # Plot por feature/batch: bar chart com p-value vs dia removido
    fig, axes = plt.subplots(
        len(FEATURES_OF_INTEREST), 2,
        figsize=(13, 2.4 * len(FEATURES_OF_INTEREST)),
        constrained_layout=True,
    )
    for i, (frontend, feature) in enumerate(FEATURES_OF_INTEREST):
        for j, batch in enumerate(("february", "april")):
            ax = axes[i, j]
            sub = [r for r in rows if r["frontend"] == frontend and r["feature"] == feature and r["batch"] == batch]
            sub.sort(key=lambda r: r["day_removed"])
            days = [r["day_removed"] for r in sub]
            ps = [r["mk_p"] for r in sub]
            xs = np.arange(len(days))
            colors = ["#2E5BFF" if p < 0.05 else "#cccccc" for p in ps]
            ax.bar(xs, ps, color=colors, edgecolor="black", linewidth=0.5)
            ax.axhline(0.05, color="red", linestyle="--", linewidth=1, label="α=0.05")
            ax.set_xticks(xs)
            ax.set_xticklabels([str(d)[-4:] for d in days], rotation=45, fontsize=8)
            ax.set_ylabel("MK p", fontsize=9)
            ax.set_title(f"{frontend}/{feature} — {batch} (removendo cada dia)",
                         fontsize=9)
            ax.set_ylim(0, max(0.3, max(ps) * 1.1))
            ax.tick_params(axis="y", labelsize=8)
            ax.grid(True, alpha=0.3, axis="y")
    fig.suptitle("011.2 — Leave-one-day-out: como cada dia afeta o p-value", fontsize=13)
    out_path = os.path.join(OUT_DIR, "leave_one_day_out.png")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[PNG] {out_path}")

    # Veredito honesto: pra cada (frontend, feature, batch), qual dia mais
    # muda o p-value vs o p original?
    print("\n=== Dias que mais MUDAM o p ===")
    for frontend, feature in FEATURES_OF_INTEREST:
        orig = next(t for t in sources[frontend]["tests"]
                    if t["feature"] == feature and t["batch"] == "april")
        p_orig = orig["tests"]["mann_kendall"]["p_value"]
        sub = [r for r in rows if r["frontend"] == frontend and r["feature"] == feature and r["batch"] == "april"]
        most_changed = max(sub, key=lambda r: abs(r["mk_p"] - p_orig))
        print(f"  [{frontend}/{feature}] april p_orig={p_orig:.3f}; "
              f"sem dia {most_changed['day_removed']}: p={most_changed['mk_p']:.3f}")


if __name__ == "__main__":
    main()
