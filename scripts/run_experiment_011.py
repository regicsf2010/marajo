"""Experimento 011 — replicar o 010 (migração de energia 0.5-5 Hz) em 60 FPS.

Roda todo o pipeline (pixel + phase) usando os vídeos /60/ em vez dos /240/
pra responder a pergunta "o detector sobrevive à mudança de frontend de captura?"

Side-effect adicional: gera 16 plots PSD (um por dia, primeiro ângulo) com o
PSD médio sobre os 10 CPs + sobreposição transparente dos 10 individuais. Esses
plots servem pra mostrar ao orientador que o sinal não é ruído branco.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import re
import sys
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig, resolve_video_path
from marajo.io.roi import load_rois
from marajo.modal.trends import all_trend_tests
from marajo.pipelines.over_time import preprocess_batch, run_over_time
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


def day_from_path(path: str) -> float:
    m = _DAY_RE.search(path)
    return float(m.group(1)) if m else 0.0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimento 011: 60 FPS.")
    p.add_argument("--config", default="configs/011-60fps.yaml")
    p.add_argument("--preprocessed-dir", default="out/all_angles_60fps/")
    p.add_argument("--cache-dir", default="out/phase_cache_60fps/")
    p.add_argument("--plots-dir", default="out/experimentos/011/")
    p.add_argument("--n-scales", type=int, default=3)
    p.add_argument("--n-orientations", type=int, default=2)
    p.add_argument("--subsample-factor", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--skip-pixel", action="store_true")
    p.add_argument("--skip-phase", action="store_true")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Pula pré-processamento (assume out/all_angles_60fps já populado).")
    p.add_argument("--skip-psd-plots", action="store_true")
    return p.parse_args()


def add_ratio_feature(
    result: TrendAnalysisResult,
    name: str,
    num_feat: str,
    den_feat: str,
    x_extractor: Callable[[str], float] | None,
    eps: float = 1e-12,
) -> None:
    """Adiciona feature derivada `name = num_feat / den_feat`. Mesma lógica do 010."""
    num_series = result.series_by_feature[num_feat]
    den_series = result.series_by_feature[den_feat]

    by_batch: dict[str, list[float]] = {}
    x_by_batch: dict[str, list[float]] = {}
    for batch in num_series.by_batch:
        num_vals = np.asarray(num_series.by_batch[batch], dtype=float)
        den_vals = np.asarray(den_series.by_batch[batch], dtype=float)
        den_safe = np.where(np.abs(den_vals) < eps, np.nan, den_vals)
        ratio = num_vals / den_safe
        by_batch[batch] = [float(v) for v in ratio]
        x_by_batch[batch] = list(num_series.x_by_batch.get(batch, []))

    new_series = FeatureSeries(feature=name, by_batch=by_batch, x_by_batch=x_by_batch)
    result.series_by_feature[name] = new_series
    result.feature_names.append(name)

    for batch, values in by_batch.items():
        x_vals = x_by_batch[batch]
        finite = np.isfinite(values)
        clean_y = [v for v, ok in zip(values, finite) if ok]
        clean_x = [x for x, ok in zip(x_vals, finite) if ok]
        tests = all_trend_tests(clean_y, x=clean_x if x_extractor is not None else None)
        result.trends.append(
            FeatureTrend(
                feature=name, batch=batch, values=values, x_values=x_vals, tests=tests
            )
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
            any_sig = any(t.p_value < alpha for t in tr.tests.values())
            flag = "OK" if any_sig else "--"
            lines.append(
                f"| `{feat}` | {batch} | {mk.statistic:+.3f} | {mk.p_value:.3f} "
                f"| {sp.statistic:+.3f} | {sp.p_value:.3f} "
                f"| {ln.slope:+.4f} | {ln.p_value:.3f} | {flag} |"
            )
    return "\n".join(lines)


def _format_detectors(result: TrendAnalysisResult, alpha: float) -> str:
    detectors: list[str] = []
    for feat in result.feature_names:
        sep = result.separates(feat, alpha=alpha)
        if sep.get("february") and not sep.get("april"):
            detectors.append(feat)
    if not detectors:
        return f"_Nenhuma feature satisfaz o criterio (tendencia em february E sem tendencia em april) com alpha = {alpha}._"
    lines = [f"Detectores candidatos (alpha = {alpha}):"]
    for feat in detectors:
        feb = result.trend(feat, "february")
        apr = result.trend(feat, "april")
        lines.append(
            f"- `{feat}`: feb MK p={feb.tests['mann_kendall'].p_value:.3f} (tau={feb.tests['mann_kendall'].statistic:+.2f}); "
            f"apr MK p={apr.tests['mann_kendall'].p_value:.3f}"
        )
    return "\n".join(lines)


def _save_raw_results(
    result: TrendAnalysisResult, alpha: float, out_dir: str, frontend: str, meta: dict
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, f"results_{frontend}.csv")
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

    json_path = os.path.join(out_dir, f"results_{frontend}.json")
    payload = {
        "frontend": frontend,
        "meta": meta,
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


def run_frontend(
    frontend: str,
    result: TrendAnalysisResult,
    args: argparse.Namespace,
    meta: dict,
) -> None:
    add_ratio_feature(
        result,
        name="frac_ratio_low_over_high_mean",
        num_feat="energy_fraction_0.5_2.0_mean",
        den_feat="energy_fraction_2.0_5.0_mean",
        x_extractor=day_from_path,
    )
    add_ratio_feature(
        result,
        name="frac_ratio_low_over_high_median",
        num_feat="energy_fraction_0.5_2.0_median",
        den_feat="energy_fraction_2.0_5.0_median",
        x_extractor=day_from_path,
    )

    plot_feature_trends(
        result, alpha=args.alpha,
        save_path=os.path.join(args.plots_dir, f"feature_trends_{frontend}.png"),
    )
    plot_pvalue_heatmap(
        result, alpha=args.alpha,
        save_path=os.path.join(args.plots_dir, f"pvalue_heatmap_{frontend}.png"),
    )

    print(f"\n## Frontend: {frontend}\n")
    print(_format_table(result, args.alpha))
    print(f"\n### Detectores candidatos ({frontend})\n")
    print(_format_detectors(result, args.alpha))

    _save_raw_results(result, args.alpha, args.plots_dir, frontend, meta)


def _psd_from_fft(fft_data):
    """Calcula |F|^2 e o eixo de frequência só pra freqs positivas."""
    psds = []
    freqs_ref = None
    for cp_id in sorted(fft_data.keys()):
        comp = fft_data[cp_id]
        psd = np.abs(comp.values) ** 2
        mask = comp.freqs > 0
        if freqs_ref is None:
            freqs_ref = comp.freqs[mask]
        psds.append(psd[mask])
    return freqs_ref, np.asarray(psds)


def plot_psd_for_day(over_time_result, day: int, save_path: str, fmax: float = 10.0) -> bool:
    """Plota PSD do primeiro vídeo do dia (10 CPs sobrepostos + médio).

    Retorna True se gerou, False se não achou vídeo do dia.
    """
    candidates = [v for v in over_time_result.video_order if f"{day}" in v]
    if not candidates:
        return False
    target = candidates[0]
    res = over_time_result.per_video[target]
    fft_data = res.fft_data
    if fft_data is None:
        return False

    freqs, psd_matrix = _psd_from_fft(fft_data)
    band_mask = freqs <= fmax
    freqs_b = freqs[band_mask]
    psd_b = psd_matrix[:, band_mask]
    psd_mean = psd_b.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(psd_b.shape[0]):
        ax.plot(freqs_b, psd_b[i], color="#888888", linewidth=0.7, alpha=0.45)
    ax.plot(freqs_b, psd_mean, color="#2E5BFF", linewidth=2.2, label="media sobre 10 CPs")
    ax.axvspan(0.5, 2.0, alpha=0.12, color="#FF8533",
               label="banda 0.5-2 Hz (detector)")
    ax.axvspan(2.0, 5.0, alpha=0.08, color="#2E5BFF",
               label="banda 2-5 Hz (phase)")

    top_idx = np.argsort(psd_mean)[-5:][::-1]
    ax.scatter(freqs_b[top_idx], psd_mean[top_idx], color="red", s=42, zorder=5,
               label="top 5 peaks (media)")

    ax.set_xlabel("frequencia (Hz)", fontsize=12)
    ax.set_ylabel("|F|^2", fontsize=12)
    ax.set_title(f"PSD - dia {day} (1o angulo) - 60 FPS, banda 0-{int(fmax)} Hz", fontsize=12)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(0, fmax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return True


def generate_psd_plots(over_time_result, plots_dir: str, frontend: str) -> None:
    out_subdir = os.path.join(plots_dir, f"psds_{frontend}")
    os.makedirs(out_subdir, exist_ok=True)

    days = sorted({int(re.search(r"(\d{8})", v).group(1)) for v in over_time_result.video_order})
    print(f"  PSDs por dia ({frontend}): {len(days)} dias")
    for d in days:
        path = os.path.join(out_subdir, f"psd_{frontend}_{d}.png")
        ok = plot_psd_for_day(over_time_result, d, path)
        if ok:
            print(f"    OK {path}")


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)
    os.makedirs(args.plots_dir, exist_ok=True)

    print(f"FPS: {config.preprocess.fps}, num_frames: {config.preprocess.num_frames}")
    print(f"Bandas: {config.modal.bands}")
    print(f"Frontends: pixel={'skip' if args.skip_pixel else 'run'} "
          f"phase={'skip' if args.skip_phase else 'run'}")
    print()

    if not args.skip_preprocess:
        print("=== Pre-processamento (60 FPS) ===")
        rois = load_rois(config.paths.rois_json)
        all_paths = [
            resolve_video_path(p, config.paths.videos_root)
            for p in (list(config.batches.february) + list(config.batches.april))
        ]
        preprocess_batch(all_paths, args.preprocessed_dir, rois, config)
        gc.collect()

    if not args.skip_pixel:
        print("\n=== Frontend pixel-grayscale ===")
        over_time = run_over_time(
            config=config,
            out_dir=args.preprocessed_dir,
            do_preprocess=False,
            keep_fft_data=True,
        )
        result = analyse_trends(over_time, config, x_extractor=day_from_path)
        run_frontend("pixel", result, args, meta={"config": args.config})

        if not args.skip_psd_plots:
            print("\n--- Gerando PSDs por dia (pixel) ---")
            generate_psd_plots(over_time, args.plots_dir, "pixel")

        del over_time, result
        gc.collect()

    if not args.skip_phase:
        print("\n=== Frontend phase-based ===")
        phase_cfg = PhaseConfig(
            n_scales=args.n_scales,
            n_orientations=args.n_orientations,
            subsample_factor=args.subsample_factor,
            use_phase_velocity=True,
        )
        over_time = run_over_time_phase_based(
            config=config,
            preprocessed_dir=args.preprocessed_dir,
            cache_dir=args.cache_dir,
            phase_config=phase_cfg,
        )
        result = analyse_trends(over_time, config, x_extractor=day_from_path)
        run_frontend(
            "phase", result, args,
            meta={"config": args.config, "phase_config": phase_cfg.__dict__},
        )

        if not args.skip_psd_plots:
            print("\n--- Gerando PSDs por dia (phase) ---")
            generate_psd_plots(over_time, args.plots_dir, "phase")


if __name__ == "__main__":
    main()
