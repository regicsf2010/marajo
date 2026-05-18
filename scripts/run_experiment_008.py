"""Experimento 008 — Phase-Based Motion Magnification como frontend.

Completa o passo que o pipeline baseline pulou: extrair movimento sub-pixel via
fase de pirâmide complexa (Wadhwa 2013, Yang 2017) ANTES de PCA+CP+FFT. O resto
do pipeline modal segue igual.

Compara diretamente com o detector pixel-grayscale (config 004-2-fine-bands):
- Mesmas sub-bandas finas (0.5-2, 2-5, 5-10, 10-20, 20-30 Hz).
- Mesmos testes de tendência (Mann-Kendall, Spearman, regressão linear).
- Mesma estrutura de saída (CSV + JSON em out/experimentos/008/).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.pipelines.phase_based import run_over_time_phase_based
from marajo.pipelines.trend_analysis import analyse_trends
from marajo.preprocessing.phase_pyramid import PhaseConfig
from marajo.viz.trends import plot_feature_trends, plot_pvalue_heatmap


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> float:
    m = _DAY_RE.search(path)
    return float(m.group(1)) if m else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento 008: phase-based MM como frontend.")
    p.add_argument("--config", default="configs/004-2-fine-bands.yaml",
                   help="Config de bandas (default: mesmo do detector validado em 004.2).")
    p.add_argument("--preprocessed-dir", default="out/all_angles/")
    p.add_argument("--cache-dir", default="out/phase_cache/")
    p.add_argument("--plots-dir", default="out/experimentos/008/")
    p.add_argument("--n-scales", type=int, default=3)
    p.add_argument("--n-orientations", type=int, default=2)
    p.add_argument("--subsample-factor", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def _format_table(result, alpha: float) -> str:
    batches = sorted({t.batch for t in result.trends})
    lines = [
        "| feature | batch | MK τ | MK p | Sp ρ | Sp p | Lin slope | Lin p | trend? |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for feat in result.feature_names:
        for batch in batches:
            tr = result.trend(feat, batch)
            mk, sp, ln = tr.tests["mann_kendall"], tr.tests["spearman"], tr.tests["linear"]
            any_sig = any(t.p_value < alpha for t in tr.tests.values())
            flag = "✅" if any_sig else "—"
            lines.append(
                f"| `{feat}` | {batch} | {mk.statistic:+.3f} | {mk.p_value:.3f} "
                f"| {sp.statistic:+.3f} | {sp.p_value:.3f} "
                f"| {ln.slope:+.4f} | {ln.p_value:.3f} | {flag} |"
            )
    return "\n".join(lines)


def _format_detectors(result, alpha: float) -> str:
    detectors: list[str] = []
    for feat in result.feature_names:
        sep = result.separates(feat, alpha=alpha)
        if sep.get("february") and not sep.get("april"):
            detectors.append(feat)
    if not detectors:
        return f"_Nenhuma feature satisfaz o critério (tendência em february E sem tendência em april) com α = {alpha}._"
    lines = [f"Features candidatas a detector (α = {alpha}):"]
    for feat in detectors:
        feb = result.trend(feat, "february")
        apr = result.trend(feat, "april")
        lines.append(
            f"- **`{feat}`**: february MK p={feb.tests['mann_kendall'].p_value:.3f} (τ={feb.tests['mann_kendall'].statistic:+.2f}); "
            f"april MK p={apr.tests['mann_kendall'].p_value:.3f}"
        )
    return "\n".join(lines)


def _save_raw_results(result, alpha: float, out_dir: str, args) -> None:
    """Persiste tudo em CSV + JSON pra análise post-hoc."""
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "batch", "test", "statistic", "p_value", "slope", "significant"])
        for t in result.trends:
            for test_name, tr in t.tests.items():
                writer.writerow([
                    t.feature, t.batch, test_name,
                    f"{tr.statistic:.6f}", f"{tr.p_value:.6f}",
                    f"{tr.slope:.6f}" if tr.slope is not None else "",
                    "1" if tr.p_value < alpha else "0",
                ])

    json_path = os.path.join(out_dir, "results.json")
    payload = {
        "config_path": args.config,
        "frontend": "phase_based_motion_magnification",
        "phase_config": {
            "n_scales": args.n_scales,
            "n_orientations": args.n_orientations,
            "subsample_factor": args.subsample_factor,
        },
        "alpha": alpha,
        "feature_names": result.feature_names,
        "series": {
            feat: {
                batch: {"x": s.x_by_batch.get(batch, []), "y": s.by_batch[batch]}
                for batch in s.by_batch
            }
            for feat, s in result.series_by_feature.items()
        },
        "tests": [
            {
                "feature": t.feature, "batch": t.batch,
                "values": t.values, "x_values": t.x_values,
                "tests": {
                    name: {"statistic": tr.statistic, "p_value": tr.p_value, "slope": tr.slope}
                    for name, tr in t.tests.items()
                },
            }
            for t in result.trends
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)

    print(f"\n[dados brutos] {csv_path}")
    print(f"[dados brutos] {json_path}")


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)
    os.makedirs(args.plots_dir, exist_ok=True)

    phase_cfg = PhaseConfig(
        n_scales=args.n_scales,
        n_orientations=args.n_orientations,
        subsample_factor=args.subsample_factor,
        use_phase_velocity=True,
    )

    print(f"Phase config: {phase_cfg}")
    print(f"Bandas: {config.modal.bands}")
    print()

    over_time = run_over_time_phase_based(
        config=config,
        preprocessed_dir=args.preprocessed_dir,
        cache_dir=args.cache_dir,
        phase_config=phase_cfg,
    )

    result = analyse_trends(over_time, config, x_extractor=day_from_path)

    plot_feature_trends(
        result, alpha=args.alpha,
        save_path=os.path.join(args.plots_dir, "feature_trends.png"),
    )
    plot_pvalue_heatmap(
        result, alpha=args.alpha,
        save_path=os.path.join(args.plots_dir, "pvalue_heatmap.png"),
    )

    print("\n## Tabela de resultados\n")
    print(_format_table(result, args.alpha))
    print("\n## Detectores candidatos\n")
    print(_format_detectors(result, args.alpha))

    _save_raw_results(result, args.alpha, args.plots_dir, args)


if __name__ == "__main__":
    main()
