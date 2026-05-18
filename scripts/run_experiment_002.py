"""Experimento 002 — features espectrais + testes de tendência.

Roda over_time (reusando vídeos pré-processados de out/w_wo_water/ por default),
extrai features espectrais por vídeo, aplica Mann-Kendall + Spearman + regressão
linear em cada (feature, batch). Gera plots em out/experimentos/002/ e imprime
tabela markdown no stdout pra colar em claude/experimentos/002-*.md.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.pipelines.over_time import run_over_time
from marajo.pipelines.trend_analysis import analyse_trends
from marajo.viz.trends import plot_feature_trends, plot_pvalue_heatmap


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento 002: features espectrais + tendência.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--out-dir", default="out/w_wo_water/", help="Onde estão os mp4 pré-processados.")
    p.add_argument("--plots-dir", default="out/experimentos/002/")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Assume que os vídeos em --out-dir já foram pré-processados.")
    p.add_argument("--freq-min", type=float, default=0.0,
                   help="Frequência mínima a considerar (descarta DC sempre; >0 também filtra baixas).")
    p.add_argument("--top-k", type=int, default=5,
                   help="K para top_k_freq_mean e energy_concentration.")
    p.add_argument("--alpha", type=float, default=0.05, help="Nível de significância.")
    return p.parse_args()


def _format_table(result, alpha: float) -> str:
    """Tabela markdown: feature × batch × test → p-value (com flag de significância)."""
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
    """Lista features que separaram (tendência em february, no-trend em april)."""
    detectors: list[str] = []
    for feat in result.feature_names:
        sep = result.separates(feat, alpha=alpha)
        if sep.get("february") and not sep.get("april"):
            detectors.append(feat)

    if not detectors:
        return "_Nenhuma feature satisfaz o critério (tendência em february E sem tendência em april) com α = {:.2f}._".format(alpha)

    lines = [f"Features candidatas a detector (α = {alpha}):"]
    for feat in detectors:
        feb = result.trend(feat, "february")
        apr = result.trend(feat, "april")
        lines.append(
            f"- **`{feat}`**: february MK p={feb.tests['mann_kendall'].p_value:.3f} (τ={feb.tests['mann_kendall'].statistic:+.2f}); "
            f"april MK p={apr.tests['mann_kendall'].p_value:.3f}"
        )
    return "\n".join(lines)


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
    )

    plot_feature_trends(
        result,
        alpha=args.alpha,
        save_path=os.path.join(args.plots_dir, "feature_trends.png"),
    )
    plot_pvalue_heatmap(
        result,
        alpha=args.alpha,
        save_path=os.path.join(args.plots_dir, "pvalue_heatmap.png"),
    )

    print("\n## Tabela de resultados\n")
    print(_format_table(result, args.alpha))
    print("\n## Detectores candidatos\n")
    print(_format_detectors(result, args.alpha))


if __name__ == "__main__":
    main()
