"""Orquestrador da análise temporal (replica main_over_time.ipynb).

Difere do baseline em dois pontos onde o notebook original tinha bugs latentes:
  - usa `video_name` (não `video_file`) ao chamar video_status;
  - lida explicitamente com o retorno de `run_cp_on_components` (que retorna mais
    de um valor; no notebook era atribuído como se fosse só um).

Estratégia de memória: por padrão NÃO acumula `SingleVideoResult` cheio por vídeo
(o `pca.score` escala com n_pixels × n_pcs e pesa centenas de MB por vídeo).
Em vez disso, guarda só `CompactVideoResult` com video_info + peaks_info, mais
opcionalmente o `unmixed` de um vídeo específico (pra espectrograma).
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from marajo.config import PipelineConfig, resolve_video_path
from marajo.io.roi import ROI, load_rois
from marajo.io.video import VideoInfo
from marajo.modal.peaks import PeakInfo
from marajo.pipelines.single_video import SingleVideoResult, analyse_preprocessed_video
from marajo.preprocessing.video_prep import PreprocessConfig, preprocess_video


@dataclass
class CompactVideoResult:
    video_info: VideoInfo
    peaks_info: dict[int, PeakInfo]
    unmixed: Optional[np.ndarray] = None  # preservado apenas pra espectrograma sob demanda


@dataclass
class OverTimeResult:
    per_video: dict[str, CompactVideoResult | SingleVideoResult]
    batches: dict[str, list[str]]                     # batch_name -> lista de video paths
    highest_freq_per_video: dict[str, list[float]]    # video -> lista de freqs (uma por CP)
    freq_per_component: dict[int, list[float]]        # CP -> lista de freqs (uma por video, na ordem global)
    mean_freqs: list[float]                           # média sobre CPs, por video (ordem global)
    mean_freqs_by_batch: dict[str, list[float]]       # batch_name -> média por video do batch
    video_order: list[str]                            # ordem global (february → april)


def preprocess_batch(
    video_paths: list[str],
    out_dir: str,
    rois: dict[str, ROI],
    config: PipelineConfig,
    skip_if_exists: bool = True,
    log: bool = True,
) -> list[str]:
    """Pré-processa cada vídeo da lista; retorna os caminhos de saída."""
    os.makedirs(out_dir, exist_ok=True)
    pre_cfg = PreprocessConfig(
        num_frames=config.preprocess.num_frames,
        fps=config.preprocess.fps,
        scale=config.preprocess.scale,
    )

    out_paths: list[str] = []
    for path in video_paths:
        name = os.path.basename(path)
        out_path = os.path.join(out_dir, name)
        roi = rois.get(name)
        if skip_if_exists and os.path.exists(out_path):
            if log:
                print(f"{path} skip (já pré-processado)")
        else:
            if log:
                print(f"{path} processing...", end=" ")
            preprocess_video(in_path=path, out_path=out_path, roi=roi, config=pre_cfg)
            if log:
                print("done.")
        out_paths.append(out_path)
    return out_paths


def run_over_time(
    config: PipelineConfig,
    out_dir: str,
    rois_json: str | None = None,
    do_preprocess: bool = True,
    keep_full: bool = False,
    keep_unmixed_for: Optional[str] = None,
) -> OverTimeResult:
    """Roda análise temporal sobre os batches definidos em `config.batches`.

    Mantém ordem global february → april. Cada vídeo é processado uma única vez;
    a divisão por batch entra apenas como metadado no resultado.

    `keep_full=False` (default): mantém só CompactVideoResult por vídeo — segura
    a memória pra processar 16+ vídeos sob WSL sem estourar.

    `keep_unmixed_for=<caminho>`: se setado, preserva `cp.unmixed` desse vídeo
    específico (útil pra espectrograma sem precisar reprocessar).
    """
    batches: dict[str, list[str]] = {
        "february": list(config.batches.february),
        "april": list(config.batches.april),
    }
    video_paths = batches["february"] + batches["april"]
    if not video_paths:
        raise ValueError("Nenhum vídeo definido em config.batches.")

    rois_json = rois_json or config.paths.rois_json
    rois = load_rois(rois_json)

    resolved = [resolve_video_path(p, config.paths.videos_root) for p in video_paths]

    if do_preprocess:
        preprocessed = preprocess_batch(resolved, out_dir, rois, config)
    else:
        preprocessed = [os.path.join(out_dir, os.path.basename(p)) for p in resolved]

    per_video: dict[str, CompactVideoResult | SingleVideoResult] = {}
    highest: dict[str, list[float]] = {}

    for original, pp in zip(video_paths, preprocessed):
        full = analyse_preprocessed_video(pp, config)
        highest[original] = [info.highest_freq for info in full.peaks_info.values()]

        if keep_full:
            per_video[original] = full
        else:
            unmixed = full.cp.unmixed.copy() if original == keep_unmixed_for else None
            per_video[original] = CompactVideoResult(
                video_info=full.video_info,
                peaks_info=full.peaks_info,
                unmixed=unmixed,
            )
            del full
            gc.collect()

    n_pcs = config.decomposition.num_pcs
    freq_per_component: dict[int, list[float]] = {i: [] for i in range(n_pcs)}
    for path in video_paths:
        for i, freq in enumerate(highest[path]):
            freq_per_component[i].append(freq)

    mean_freqs = [float(np.mean(highest[p])) for p in video_paths]
    mean_freqs_by_batch = {
        name: [float(np.mean(highest[p])) for p in paths]
        for name, paths in batches.items()
    }

    return OverTimeResult(
        per_video=per_video,
        batches=batches,
        highest_freq_per_video=highest,
        freq_per_component=freq_per_component,
        mean_freqs=mean_freqs,
        mean_freqs_by_batch=mean_freqs_by_batch,
        video_order=list(video_paths),
    )
