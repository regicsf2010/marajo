"""012.2 — Reconciliação benchmark antigo (fev/abr) × campanha pareada nova.

Post-hoc puro sobre brutos existentes (010 = dataset antigo; 012 = campanha 3).
Não reprocessa vídeo. Compara o SINAL de τ (Mann-Kendall) das mesmas features entre:

  Lado estresse:  february (antigo, estresse leve/moderado)  vs  stressed_m / stressed_n (novo, dessecação severa)
  Lado controle:  april (antigo)                              vs  control_m / control_n (novo)

Hipótese (não-monotonicidade, Sánchez-López 2020): conforme a severidade aumenta de
turgor-loss → dessecação, o sinal de τ vira. O antigo e o novo são pontos diferentes da
mesma curva, não uma contradição.

Saídas em out/experimentos/012-2/:
- reconcile_tau.csv — τ e p de cada feature/condição, com flag de flip de sinal.
- reconcile_heatmap.png/pdf — heatmaps de τ (pixel/phase × estresse/controle).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Features fisicamente relevantes (band energy + centroide). peak_freq foi quase sempre nulo.
FEATURES = [
    "energy_0.5_2.0_mean", "energy_0.5_2.0_median",
    "energy_2.0_5.0_mean", "energy_2.0_5.0_median",
    "energy_0.5_5.0_mean", "energy_0.5_5.0_median",
    "centroid_2.0_5.0_mean", "centroid_2.0_5.0_median",
    "centroid_0.5_5.0_mean", "centroid_0.5_5.0_median",
]

STRESS_COLS = [("antigo (010)", "february"), ("nova manhã", "stressed_m"), ("nova noite", "stressed_n")]
CONTROL_COLS = [("antigo (010)", "april"), ("nova manhã", "control_m"), ("nova noite", "control_n")]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="012.2 — reconciliação fev/abr × campanha pareada.")
    p.add_argument("--old-dir", default="out/experimentos/010")
    p.add_argument("--new-dir", default="out/experimentos/012")
    p.add_argument("--out-dir", default="out/experimentos/012-2")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_mk(path: str) -> dict[tuple[str, str], tuple[float, float]]:
    """{(feature, batch): (tau, p)} a partir do JSON de resultados."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for t in data["tests"]:
        mk = t["tests"]["mann_kendall"]
        out[(t["feature"], t["batch"])] = (mk["statistic"], mk["p_value"])
    return out


def gather(old_mk, new_mk, cols) -> tuple[np.ndarray, np.ndarray]:
    """Matrizes [feature × coluna] de τ e p. Coluna 0 vem do antigo, 1-2 do novo."""
    tau = np.full((len(FEATURES), len(cols)), np.nan)
    pval = np.full((len(FEATURES), len(cols)), np.nan)
    for i, feat in enumerate(FEATURES):
        for j, (_, batch) in enumerate(cols):
            src = old_mk if j == 0 else new_mk
            if (feat, batch) in src:
                tau[i, j], pval[i, j] = src[(feat, batch)]
    return tau, pval


def draw_heatmap(ax, tau, pval, cols, title, alpha):
    im = ax.imshow(tau, cmap="RdBu_r", vmin=-0.8, vmax=0.8, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[0] for c in cols], fontsize=8)
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels(FEATURES, fontsize=7)
    ax.set_title(title, fontsize=10, weight="bold")
    for i in range(tau.shape[0]):
        for j in range(tau.shape[1]):
            if np.isnan(tau[i, j]):
                continue
            sig = "*" if pval[i, j] < alpha else ""
            txt = f"{tau[i, j]:+.2f}{sig}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    weight="bold" if sig else "normal",
                    color="white" if abs(tau[i, j]) > 0.45 else "black")
    return im


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rows_csv = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 11), constrained_layout=True)
    im = None
    for r, frontend in enumerate(("pixel", "phase")):
        old_mk = load_mk(os.path.join(args.old_dir, f"results_{frontend}.json"))
        new_mk = load_mk(os.path.join(args.new_dir, f"results_{frontend}.json"))
        for c, (side, cols) in enumerate([("estresse", STRESS_COLS), ("controle", CONTROL_COLS)]):
            tau, pval = gather(old_mk, new_mk, cols)
            im = draw_heatmap(axes[r, c], tau, pval, cols,
                              f"{frontend} · {side}  (τ Mann-Kendall, * = p<{args.alpha})", args.alpha)
            for i, feat in enumerate(FEATURES):
                for j, (cname, batch) in enumerate(cols):
                    rows_csv.append([frontend, side, feat, cname, batch,
                                     f"{tau[i,j]:.4f}" if not np.isnan(tau[i,j]) else "",
                                     f"{pval[i,j]:.4f}" if not np.isnan(pval[i,j]) else ""])
            # flip: sinal de τ antigo (col 0) vs nova manhã (col 1), ambos significativos
            if side == "estresse":
                for i, feat in enumerate(FEATURES):
                    t_old, p_old = tau[i, 0], pval[i, 0]
                    t_new, p_new = tau[i, 1], pval[i, 1]
                    if not np.isnan(t_old) and not np.isnan(t_new) and np.sign(t_old) != np.sign(t_new):
                        if p_old < 0.15 and p_new < args.alpha:
                            print(f"FLIP {frontend} {feat}: antigo τ={t_old:+.2f}(p={p_old:.3f}) "
                                  f"→ nova manhã τ={t_new:+.2f}(p={p_new:.3f})")

    cbar = fig.colorbar(im, ax=axes, shrink=0.6, location="right")
    cbar.set_label("τ (Mann-Kendall) — vermelho sobe, azul desce", fontsize=9)
    fig.suptitle("012.2 — Reconciliação: benchmark antigo (estresse leve) × campanha pareada (dessecação severa)",
                 fontsize=12, weight="bold")

    csv_path = os.path.join(args.out_dir, "reconcile_tau.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frontend", "side", "feature", "condicao", "batch", "tau", "p_value"])
        w.writerows(rows_csv)
    print(f"\n[dados brutos] {csv_path}")
    for ext in ("png", "pdf"):
        p = os.path.join(args.out_dir, f"reconcile_heatmap.{ext}")
        fig.savefig(p, dpi=130, bbox_inches="tight")
        print(f"  {p}")
    plt.close(fig)


if __name__ == "__main__":
    main()
