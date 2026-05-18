"""Experimento 003 — 4 ângulos/dia como observações independentes.

Roda over_time com configs/003-all-angles.yaml (64 vídeos = 8 dias × 4 ângulos × 2
batches), aplica os mesmos testes de tendência do 002 mas com x = dia do calendário
(extraído do path via regex YYYYMMDD). Pares com mesmo x são ignorados no MK.

Pré-condição: a ROI do "principal" cada dia serve pros 4 ângulos (câmera fixa).
Resolvido em `marajo.io.roi.resolve_roi`.
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
from marajo.pipelines.over_time import run_over_time
from marajo.pipelines.trend_analysis import analyse_trends
from marajo.viz.trends import plot_feature_trends, plot_pvalue_heatmap


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> float:
    """Extrai YYYYMMDD do path como inteiro pra usar como x nos testes."""
    m = _DAY_RE.search(path)
    if not m:
        raise ValueError(f"Não consegui extrair YYYYMMDD de {path!r}")
    return float(m.group(1))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento 003: 4 ângulos/dia + testes de tendência.")
    p.add_argument("--config", default="configs/003-all-angles.yaml")
    p.add_argument("--out-dir", default="out/all_angles/",
                   help="Onde salvar / buscar os mp4 pré-processados.")
    p.add_argument("--plots-dir", default="out/experimentos/003/")
    p.add_argument("--no-preprocess", action="store_true")
    p.add_argument("--freq-min", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=5)
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


def _save_raw_results(result, alpha: float, out_dir: str, config_path: str) -> None:
    """Persiste resultados brutos em CSV (uma linha por feature × batch × teste) +
    JSON com séries completas (valores e x por feature × batch).

    O CSV serve pra leitura programática direta (pandas, R, Excel). O JSON guarda
    os dados originais pra qualquer análise post-hoc sem reprocessar.
    """
    os.makedirs(out_dir, exist_ok=True)

    # CSV: feature, batch, test, statistic, p_value, slope, significant
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

    # JSON: estrutura completa com séries originais
    json_path = os.path.join(out_dir, "results.json")
    payload = {
        "config_path": config_path,
        "alpha": alpha,
        "freq_min": result.freq_min,
        "top_k": result.top_k,
        "feature_names": result.feature_names,
        "series": {
            feat: {
                batch: {
                    "x": series.x_by_batch.get(batch, []),
                    "y": series.by_batch[batch],
                }
                for batch in series.by_batch
            }
            for feat, series in result.series_by_feature.items()
        },
        "tests": [
            {
                "feature": t.feature,
                "batch": t.batch,
                "values": t.values,
                "x_values": t.x_values,
                "tests": {
                    name: {
                        "statistic": tr.statistic,
                        "p_value": tr.p_value,
                        "slope": tr.slope,
                    }
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

    over_time = run_over_time(
        config=config,
        out_dir=args.out_dir,
        do_preprocess=not args.no_preprocess,
        keep_fft_data=True,
    )

    result = analyse_trends(
        over_time_result=over_time,
        config=config,
        freq_min=args.freq_min,
        top_k=args.top_k,
        x_extractor=day_from_path,
    )

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

    _save_raw_results(result, args.alpha, args.plots_dir, args.config)


if __name__ == "__main__":
    main()
