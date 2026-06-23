"""Experimento 012 — validação pareada out-of-sample na 3ª campanha (`3_camp/`).

Design PAREADO: 2 plantas filmadas nos mesmos 14 dias, mesma câmera, mesmos
horários (manhã `m` + noite `n`). `_1` = controle regado, `_2` = estressada
(déficit). Isso elimina os confounds de câmera/época que dominaram o benchmark
antigo (fevereiro/G60 vs abril/Signature).

Reusa a análise do 010 (bandas 0.5-2, 2-5, 0.5-5 + razão low/high) nos 2 frontends
(pixel-grayscale e phase-based). Critério de detector, por período X ∈ {m, n}:
a feature deve ter tendência em `stressed_X` (p < α) e NÃO em `control_X` (p > α).

Diferente dos experimentos anteriores, aqui os vídeos são NOVOS → o frontend pixel
roda com do_preprocess=True (popula o dir de pré-processados) e o phase lê de lá.

Pré-requisito: ROIs das 56 capturas em rois/rois.json (rode setup_3camp_rois.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.modal.trends import all_trend_tests
from marajo.pipelines.over_time import run_over_time
from marajo.pipelines.phase_based import run_over_time_phase_based
from marajo.pipelines.trend_analysis import (
    FeatureSeries,
    FeatureTrend,
    TrendAnalysisResult,
    analyse_trends,
)
from marajo.preprocessing.phase_pyramid import PhaseConfig
from marajo.viz.trends import plot_feature_trends, plot_pvalue_heatmap

_DAY_RE = re.compile(r"(\d{8})")
PERIODS = ("m", "n")  # manhã, noite


def day_from_path(path: str) -> float:
    """Primeiro YYYYMMDD do path = dia experimental (pasta do dia, não o timestamp
    do nome — vídeos de noite às vezes cruzam a meia-noite)."""
    m = _DAY_RE.search(path)
    return float(m.group(1)) if m else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento 012: validação pareada (3ª campanha).")
    p.add_argument("--config", default="configs/012-paired-campaign.yaml")
    p.add_argument("--preprocessed-dir", default="out/3camp_preprocessed/")
    p.add_argument("--cache-dir", default="out/phase_cache_3camp/")
    p.add_argument("--plots-dir", default="out/experimentos/012/")
    p.add_argument("--n-scales", type=int, default=3)
    p.add_argument("--n-orientations", type=int, default=2)
    p.add_argument("--subsample-factor", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--skip-pixel", action="store_true")
    p.add_argument("--skip-phase", action="store_true")
    p.add_argument("--no-preprocess", action="store_true",
                   help="reusar pré-processados existentes em --preprocessed-dir")
    return p.parse_args()


def add_ratio_feature(
    result: TrendAnalysisResult, name: str, num_feat: str, den_feat: str,
    x_extractor: Callable[[str], float] | None, eps: float = 1e-12,
) -> None:
    """Adiciona feature derivada `name = num_feat / den_feat` e roda os 3 testes."""
    num_series = result.series_by_feature[num_feat]
    den_series = result.series_by_feature[den_feat]
    by_batch: dict[str, list[float]] = {}
    x_by_batch: dict[str, list[float]] = {}
    for batch in num_series.by_batch:
        num_vals = np.asarray(num_series.by_batch[batch], dtype=float)
        den_vals = np.asarray(den_series.by_batch[batch], dtype=float)
        den_safe = np.where(np.abs(den_vals) < eps, np.nan, den_vals)
        by_batch[batch] = [float(v) for v in (num_vals / den_safe)]
        x_by_batch[batch] = list(num_series.x_by_batch.get(batch, []))
    result.series_by_feature[name] = FeatureSeries(feature=name, by_batch=by_batch, x_by_batch=x_by_batch)
    result.feature_names.append(name)
    for batch, values in by_batch.items():
        x_vals = x_by_batch[batch]
        finite = np.isfinite(values)
        clean_y = [v for v, ok in zip(values, finite) if ok]
        clean_x = [x for x, ok in zip(x_vals, finite) if ok]
        tests = all_trend_tests(clean_y, x=clean_x if x_extractor is not None else None)
        result.trends.append(
            FeatureTrend(feature=name, batch=batch, values=values, x_values=x_vals, tests=tests)
        )


def _format_table(result: TrendAnalysisResult, alpha: float) -> str:
    batches = sorted({t.batch for t in result.trends})
    lines = [
        "| feature | batch | MK τ | MK p | Sp ρ | Sp p | Lin slope | Lin p | trend? |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for feat in result.feature_names:
        for batch in batches:
            tr = result.trend(feat, batch)
            mk, sp, ln = tr.tests["mann_kendall"], tr.tests["spearman"], tr.tests["linear"]
            flag = "✅" if any(t.p_value < alpha for t in tr.tests.values()) else "—"
            lines.append(
                f"| `{feat}` | {batch} | {mk.statistic:+.3f} | {mk.p_value:.3f} "
                f"| {sp.statistic:+.3f} | {sp.p_value:.3f} "
                f"| {ln.slope:+.4f} | {ln.p_value:.3f} | {flag} |"
            )
    return "\n".join(lines)


def _format_detectors(result: TrendAnalysisResult, alpha: float) -> str:
    """Detector pareado: tendência em stressed_X e estacionário em control_X, por período."""
    out_lines: list[str] = []
    for per in PERIODS:
        s, c = f"stressed_{per}", f"control_{per}"
        batches_present = {t.batch for t in result.trends}
        if s not in batches_present or c not in batches_present:
            continue
        dets = []
        for feat in result.feature_names:
            sep = result.separates(feat, alpha=alpha)
            if sep.get(s) and not sep.get(c):
                dets.append(feat)
        label = "manhã" if per == "m" else "noite"
        if not dets:
            out_lines.append(f"- **{label}**: nenhuma feature separa (tendência em {s}, estável em {c}) com α={alpha}.")
            continue
        out_lines.append(f"- **{label}** (α={alpha}):")
        for feat in dets:
            st = result.trend(feat, s)
            ct = result.trend(feat, c)
            out_lines.append(
                f"    - `{feat}`: {s} MK p={st.tests['mann_kendall'].p_value:.3f} "
                f"(τ={st.tests['mann_kendall'].statistic:+.2f}); {c} MK p={ct.tests['mann_kendall'].p_value:.3f}"
            )
    return "\n".join(out_lines)


def _save_raw_results(result: TrendAnalysisResult, alpha: float, out_dir: str, frontend: str, meta: dict) -> None:
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"results_{frontend}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["feature", "batch", "test", "statistic", "p_value", "slope", "significant"])
        for t in result.trends:
            for test_name, tr in t.tests.items():
                w.writerow([
                    t.feature, t.batch, test_name,
                    f"{tr.statistic:.6f}", f"{tr.p_value:.6f}",
                    f"{tr.slope:.6f}" if tr.slope is not None else "",
                    "1" if tr.p_value < alpha else "0",
                ])
    json_path = os.path.join(out_dir, f"results_{frontend}.json")
    payload = {
        "frontend": frontend, "meta": meta, "alpha": alpha,
        "feature_names": result.feature_names,
        "series": {
            feat: {b: {"x": s.x_by_batch.get(b, []), "y": s.by_batch[b]} for b in s.by_batch}
            for feat, s in result.series_by_feature.items()
        },
        "tests": [
            {
                "feature": t.feature, "batch": t.batch, "values": t.values, "x_values": t.x_values,
                "tests": {n: {"statistic": tr.statistic, "p_value": tr.p_value, "slope": tr.slope}
                          for n, tr in t.tests.items()},
            }
            for t in result.trends
        ],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\n[dados brutos] {csv_path}")
    print(f"[dados brutos] {json_path}")


def post_process(frontend: str, result: TrendAnalysisResult, args: argparse.Namespace, meta: dict) -> None:
    add_ratio_feature(result, "frac_ratio_low_over_high_mean",
                      "energy_fraction_0.5_2.0_mean", "energy_fraction_2.0_5.0_mean", day_from_path)
    add_ratio_feature(result, "frac_ratio_low_over_high_median",
                      "energy_fraction_0.5_2.0_median", "energy_fraction_2.0_5.0_median", day_from_path)
    plot_feature_trends(result, alpha=args.alpha,
                        save_path=os.path.join(args.plots_dir, f"feature_trends_{frontend}.png"))
    plot_pvalue_heatmap(result, alpha=args.alpha,
                        save_path=os.path.join(args.plots_dir, f"pvalue_heatmap_{frontend}.png"))
    print(f"\n## Frontend: {frontend}\n")
    print(_format_table(result, args.alpha))
    print(f"\n### Detectores candidatos ({frontend})\n")
    print(_format_detectors(result, args.alpha))
    _save_raw_results(result, args.alpha, args.plots_dir, frontend, meta)


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"Batches: {{ {', '.join(f'{k}: {len(v)}' for k, v in config.batches.items())} }}")
    print(f"Bandas: {config.modal.bands}")
    print(f"Frontends: pixel={'skip' if args.skip_pixel else 'run'} phase={'skip' if args.skip_phase else 'run'}")
    print(f"fps={config.preprocess.fps} num_frames={config.preprocess.num_frames} scale={config.preprocess.scale}\n")

    if not args.skip_pixel:
        print("=== Frontend pixel-grayscale ===")
        over_time = run_over_time(
            config=config,
            out_dir=args.preprocessed_dir,
            do_preprocess=not args.no_preprocess,
            keep_fft_data=True,
        )
        result = analyse_trends(over_time, config, x_extractor=day_from_path)
        post_process("pixel", result, args, meta={"config": args.config})
        del over_time, result

    if not args.skip_phase:
        print("\n=== Frontend phase-based ===")
        phase_cfg = PhaseConfig(
            n_scales=args.n_scales, n_orientations=args.n_orientations,
            subsample_factor=args.subsample_factor, use_phase_velocity=True,
        )
        over_time = run_over_time_phase_based(
            config=config,
            preprocessed_dir=args.preprocessed_dir,
            cache_dir=args.cache_dir,
            phase_config=phase_cfg,
        )
        result = analyse_trends(over_time, config, x_extractor=day_from_path)
        post_process("phase", result, args, meta={"config": args.config, "phase_config": phase_cfg.__dict__})


if __name__ == "__main__":
    main()
