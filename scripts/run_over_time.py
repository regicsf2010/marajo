"""CLI para rodar a análise temporal (replica main_over_time)."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.pipelines.over_time import run_over_time
from marajo.viz.over_time import (
    plot_batches_over_time,
    plot_components_over_time,
    plot_mean_over_time,
)
from marajo.viz.spectra import plot_spectrogram
from marajo.modal.spectrogram import compute_spectrogram


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Análise temporal sobre lista canônica de vídeos.")
    p.add_argument("--config", default="configs/default.yaml", help="YAML de configuração.")
    p.add_argument("--out-dir", default="out/w_wo_water/",
                   help="Diretório de saída dos vídeos pré-processados.")
    p.add_argument("--plots-dir", default="out/",
                   help="Diretório onde salvar os gráficos.")
    p.add_argument("--no-preprocess", action="store_true",
                   help="Assume que os vídeos já foram pré-processados em --out-dir.")
    p.add_argument("--spectrogram-video-idx", type=int, default=None,
                   help="Se setado, gera um espectrograma da fonte 1 desse índice de vídeo.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)

    keep_unmixed_for = None
    if args.spectrogram_video_idx is not None:
        all_videos = config.batches.february + config.batches.april
        keep_unmixed_for = all_videos[args.spectrogram_video_idx]

    result = run_over_time(
        config=config,
        out_dir=args.out_dir,
        do_preprocess=not args.no_preprocess,
        keep_unmixed_for=keep_unmixed_for,
    )

    os.makedirs(args.plots_dir, exist_ok=True)

    plot_components_over_time(
        result.freq_per_component,
        save_path=os.path.join(args.plots_dir, "components_over_time.png"),
    )
    plot_mean_over_time(
        result.mean_freqs,
        save_path=os.path.join(args.plots_dir, "mean_over_time.png"),
    )
    plot_batches_over_time(
        result.mean_freqs_by_batch,
        save_path=os.path.join(args.plots_dir, "pipeline_over_time.png"),
    )

    if args.spectrogram_video_idx is not None:
        video_key = result.video_order[args.spectrogram_video_idx]
        compact = result.per_video[video_key]
        unmixed = compact.unmixed  # type: ignore[union-attr]
        info = compact.video_info
        spec = compute_spectrogram(unmixed[:, 1], fps=info.fps)
        plot_spectrogram(spec, save_path=os.path.join(args.plots_dir, "spect.png"))

    plt.show()


if __name__ == "__main__":
    main()
