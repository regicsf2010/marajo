"""CLI para rodar o pipeline em um único vídeo e salvar os plots."""

from __future__ import annotations

import argparse
import os
import sys

# permite rodar como `python scripts/run_single_video.py` sem instalar o pacote
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig, resolve_video_path
from marajo.io.roi import ROI, get_roi_for_video
from marajo.pipelines.single_video import run_single_video
from marajo.viz.spectra import plot_source_psd_phase


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pipeline para um único vídeo.")
    p.add_argument("--video", required=True, help="Caminho do vídeo de entrada.")
    p.add_argument("--out", required=True, help="Caminho do vídeo pré-processado de saída (mp4).")
    p.add_argument("--config", default="configs/default.yaml", help="YAML de configuração.")
    p.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                   help="ROI explícita; se omitido, busca em rois.json pelo nome do arquivo.")
    p.add_argument("--skip-preprocess-if-exists", action="store_true",
                   help="Pula a etapa de pré-processamento se --out já existir.")
    p.add_argument("--plot-out", default=None,
                   help="Caminho pra salvar o plot PSD/phase (PDF/PNG). Se omitido, não salva.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)

    video_path = resolve_video_path(args.video, config.paths.videos_root)

    if args.roi:
        roi = ROI(*args.roi)
    else:
        roi = get_roi_for_video(video_path, json_path=config.paths.rois_json)
        if roi is None:
            print(f"[aviso] sem ROI registrada pra {os.path.basename(video_path)}; processando sem crop.")

    result = run_single_video(
        in_video_path=video_path,
        out_video_path=args.out,
        roi=roi,
        config=config,
        skip_preprocess_if_exists=args.skip_preprocess_if_exists,
    )

    print(f"Vídeo pré-processado: {result.preprocessed_path}")
    print(f"Info: fps={result.video_info.fps} frames={result.video_info.frames}")
    print("Frequências dominantes por CP:")
    for cp_id, info in result.peaks_info.items():
        print(f"  CP {cp_id}: highest={info.highest_freq:.2f} Hz amp={info.highest_amp:.2e}")

    plot_source_psd_phase(
        fft_data=result.fft_data,
        peaks_info=result.peaks_info,
        video=result.video_info,
        components=range(config.decomposition.num_pcs),
        save_path=args.plot_out,
    )


if __name__ == "__main__":
    main()
