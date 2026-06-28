"""012.1 — Feature de divergência (estressada − controle) + figura-síntese.

Post-hoc sobre os brutos do 012 (não reprocessa vídeo). Para cada feature × período:
normaliza a série de cada planta à linha de base do dia 1 (fold-change), calcula a
divergência `div_d = stressed_fold_d − control_fold_d` e roda os 3 testes de tendência
na divergência. Num design pareado de seedling em crescimento, a divergência é o detector
correto (o controle não é estacionário — ele cresce — mas diverge da estressada).

Saídas em out/experimentos/012-1/:
- divergence_results.{csv,json} — stats da tendência da divergência (todas as features).
- divergence_detectors.png/pdf — figura-síntese: tira de miniaturas da estressada
  (verde→seca, 14 dias) + 6 detectores mostrando as 2 plantas divergindo.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.modal.trends import all_trend_tests

STRESSED_ROI = (10, 130, 760, 1330)  # x, y, w, h — pra recortar a miniatura na planta

# Detectores destacados na figura (frontend, periodo, feature, título curto).
HIGHLIGHT = [
    ("phase", "m", "energy_2.0_5.0_median", "phase · manhã · energy 2–5 Hz"),
    ("phase", "m", "energy_0.5_5.0_median", "phase · manhã · energy 0.5–5 Hz"),
    ("phase", "n", "centroid_2.0_5.0_mean", "phase · noite · centroid 2–5 Hz"),
    ("pixel", "n", "energy_0.5_5.0_mean", "pixel · noite · energy 0.5–5 Hz"),
    ("pixel", "m", "centroid_2.0_5.0_median", "pixel · manhã · centroid 2–5 Hz"),
    ("pixel", "m", "centroid_0.5_5.0_mean", "pixel · manhã · centroid 0.5–5 Hz"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="012.1 — divergência estressada−controle + figura.")
    p.add_argument("--results-dir", default="out/experimentos/012")
    p.add_argument("--out-dir", default="out/experimentos/012-1")
    p.add_argument("--config", default="configs/012-paired-campaign.yaml")
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def load_series(results: dict, feature: str, batch: str) -> tuple[np.ndarray, np.ndarray]:
    """Devolve (dias_ordenados_1..n, y_ordenado) pra (feature, batch)."""
    s = results["series"][feature][batch]
    x = np.asarray(s["x"], dtype=float)
    y = np.asarray(s["y"], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def fold(y: np.ndarray) -> np.ndarray:
    """Fold-change relativo ao dia 1 (guarda contra zero)."""
    base = y[0] if abs(y[0]) > 1e-12 else (np.mean(y) if abs(np.mean(y)) > 1e-12 else 1.0)
    return y / base


def divergence_for(results: dict, feature: str, period: str) -> dict | None:
    s_batch, c_batch = f"stressed_{period}", f"control_{period}"
    if feature not in results["series"]:
        return None
    if s_batch not in results["series"][feature] or c_batch not in results["series"][feature]:
        return None
    xs, ys = load_series(results, feature, s_batch)
    xc, yc = load_series(results, feature, c_batch)
    if not np.array_equal(xs, xc):
        # dias precisam casar pro pareamento
        return None
    s_fold, c_fold = fold(ys), fold(yc)
    div = s_fold - c_fold
    day_idx = list(range(1, len(div) + 1))
    tests = all_trend_tests(list(div), x=day_idx)
    return {
        "feature": feature, "period": period, "days": [float(d) for d in xs],
        "stressed_fold": [float(v) for v in s_fold], "control_fold": [float(v) for v in c_fold],
        "divergence": [float(v) for v in div], "tests": tests,
    }


def stressed_thumbnails(config: PipelineConfig, n_days: int = 14) -> list[np.ndarray]:
    """1 frame por dia da estressada de manhã (stressed_m), recortado na ROI, RGB pequeno."""
    paths = config.batches["stressed_m"]
    x, y, w, h = STRESSED_ROI
    thumbs = []
    for rel in paths[:n_days]:
        full = os.path.join(config.paths.videos_root, rel)
        cap = cv.VideoCapture(full)
        ok, fr = cap.read()
        cap.release()
        if not ok:
            thumbs.append(np.zeros((220, 130, 3), dtype=np.uint8))
            continue
        crop = fr[y:y + h, x:x + w]
        crop = cv.cvtColor(crop, cv.COLOR_BGR2RGB)
        crop = cv.resize(crop, (130, 220))
        thumbs.append(crop)
    return thumbs


def make_figure(by_key: dict, thumbs: list[np.ndarray], days: list[float], alpha: float, out_dir: str) -> None:
    n_det = len(HIGHLIGHT)
    fig = plt.figure(figsize=(16, 12.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.85, 1.0, 1.0],
                          top=0.88, bottom=0.05, hspace=0.45, wspace=0.24)

    fig.suptitle("012.1 — Divergência estressada − controle (design pareado, campanha 3)",
                 y=0.975, fontsize=14, weight="bold")
    fig.text(0.5, 0.925, "Planta estressada (manhã) — evolução verde → dessecada ao longo de 14 dias",
             ha="center", fontsize=11)

    # --- tira de miniaturas (linha de cima, ocupa as 3 colunas via sub-gridspec) ---
    strip = gs[0, :].subgridspec(1, len(thumbs), wspace=0.05)
    for i, th in enumerate(thumbs):
        ax = fig.add_subplot(strip[0, i])
        ax.imshow(th)
        ax.set_title(f"d{i+1}", fontsize=8, pad=2)
        ax.axis("off")

    # --- detectores (2 linhas × 3 colunas) ---
    for k, (frontend, period, feat, title) in enumerate(HIGHLIGHT):
        row, col = 1 + k // 3, k % 3
        ax = fig.add_subplot(gs[row, col])
        d = by_key.get((frontend, period, feat))
        if d is None:
            ax.text(0.5, 0.5, f"{title}\n(sem dados)", ha="center", va="center")
            ax.axis("off")
            continue
        xi = list(range(1, len(d["divergence"]) + 1))
        ax.plot(xi, d["stressed_fold"], "o-", color="tab:red", label="estressada", alpha=0.85)
        ax.plot(xi, d["control_fold"], "o-", color="tab:green", label="controle", alpha=0.85)
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax2 = ax.twinx()
        ax2.fill_between(xi, d["divergence"], 0, color="tab:purple", alpha=0.15)
        ax2.plot(xi, d["divergence"], "--", color="tab:purple", lw=1.2, label="divergência")
        ax2.set_ylabel("divergência", color="tab:purple", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="tab:purple", labelsize=7)
        mk = d["tests"]["mann_kendall"]
        sp = d["tests"]["spearman"]
        flag = "✔" if mk.p_value < alpha else ""
        ax.set_title(f"{title}\ndiv: MK τ={mk.statistic:+.2f} p={mk.p_value:.3f} {flag} | Sp p={sp.p_value:.3f}",
                     fontsize=9)
        ax.set_xlabel("dia")
        ax.set_ylabel("fold-change vs dia 1", fontsize=8)
        if k == 0:
            ax.legend(loc="upper left", fontsize=7)
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"divergence_detectors.{ext}")
        fig.savefig(path, dpi=130, bbox_inches="tight")
        print(f"  {path}")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    config = PipelineConfig.load(args.config)

    results = {}
    for frontend in ("pixel", "phase"):
        with open(os.path.join(args.results_dir, f"results_{frontend}.json")) as f:
            results[frontend] = json.load(f)

    # Divergência pra TODAS as features × período, ambos frontends.
    rows = []
    by_key = {}
    for frontend, res in results.items():
        for feat in res["feature_names"]:
            for period in ("m", "n"):
                d = divergence_for(res, feat, period)
                if d is None:
                    continue
                d["frontend"] = frontend
                by_key[(frontend, period, feat)] = d
                rows.append(d)

    # CSV + JSON brutos
    csv_path = os.path.join(args.out_dir, "divergence_results.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frontend", "period", "feature", "test", "statistic", "p_value", "slope", "significant"])
        for d in rows:
            for tname, tr in d["tests"].items():
                w.writerow([d["frontend"], d["period"], d["feature"], tname,
                            f"{tr.statistic:.6f}", f"{tr.p_value:.6f}",
                            f"{tr.slope:.6f}" if tr.slope is not None else "",
                            "1" if tr.p_value < args.alpha else "0"])
    json_path = os.path.join(args.out_dir, "divergence_results.json")
    with open(json_path, "w") as f:
        json.dump([
            {**{k: d[k] for k in ("frontend", "period", "feature", "days",
                                   "stressed_fold", "control_fold", "divergence")},
             "tests": {n: {"statistic": tr.statistic, "p_value": tr.p_value, "slope": tr.slope}
                       for n, tr in d["tests"].items()}}
            for d in rows
        ], f, indent=2, default=float)
    print(f"[dados brutos] {csv_path}\n[dados brutos] {json_path}")

    # Ranking: divergências mais fortes (por MK p)
    ranked = sorted(rows, key=lambda d: d["tests"]["mann_kendall"].p_value)
    print("\n## Top divergências (estressada − controle):\n")
    print("| frontend | per | feature | MK τ | MK p | Sp p | Lin p |")
    print("|---|---|---|---:|---:|---:|---:|")
    for d in ranked[:15]:
        mk, sp, ln = d["tests"]["mann_kendall"], d["tests"]["spearman"], d["tests"]["linear"]
        print(f"| {d['frontend']} | {d['period']} | `{d['feature']}` | {mk.statistic:+.2f} "
              f"| {mk.p_value:.3f} | {sp.p_value:.3f} | {ln.p_value:.3f} |")

    # Figura-síntese
    print("\n## Figura-síntese:")
    thumbs = stressed_thumbnails(config)
    days = rows[0]["days"] if rows else []
    make_figure(by_key, thumbs, days, args.alpha, args.out_dir)


if __name__ == "__main__":
    main()
