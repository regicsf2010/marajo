"""Pipeline modal sobre sinais de fase (phase-based motion magnification).

Substitui o "dataset = frames achatados" do pipeline baseline por
"dataset = sinais de fase (n_frames-1, total_features_subband)" da pirâmide
complexa steerable. PCA + CP + FFT + features espectrais reusados do pacote.

Como cada vídeo gera ~10 MB de cache de fase, processar uma vez e reusar muitas
vezes é o que torna isso viável pra iteração.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from marajo.config import PipelineConfig
from marajo.decomposition.cp import run_cp_on_components
from marajo.decomposition.pca import compute_pca
from marajo.io.video import VideoInfo, video_status
from marajo.modal.fft import compute_fft_for_components
from marajo.modal.peaks import highest_peak_frequencies
from marajo.pipelines.over_time import CompactVideoResult, OverTimeResult
from marajo.pipelines.single_video import SingleVideoResult
from marajo.preprocessing.phase_pyramid import (
    PhaseConfig,
    compute_and_cache,
    flatten_signals,
)


def analyse_phase_video(
    preprocessed_path: str,
    cache_path: str,
    config: PipelineConfig,
    phase_config: Optional[PhaseConfig] = None,
) -> SingleVideoResult:
    """Pipeline single vídeo via phase signals."""
    phase_cfg = phase_config or PhaseConfig()

    data = compute_and_cache(preprocessed_path, cache_path, phase_cfg)
    dataset = flatten_signals(data)  # (n_eff_frames, n_features)
    n_eff = dataset.shape[0]

    info_raw = video_status(preprocessed_path)
    info_eff = replace(info_raw, frames=n_eff)

    pca = compute_pca(dataset, n_components=config.decomposition.num_pcs)
    del dataset

    cp = run_cp_on_components(pca, n_pc=config.decomposition.num_pcs, config=config.cp)

    fft_data = compute_fft_for_components(
        cp.unmixed, info_eff.fps, range(config.decomposition.num_pcs)
    )
    peaks = highest_peak_frequencies(fft_data, n_peaks=config.modal.n_peaks)

    return SingleVideoResult(
        video_info=info_eff,
        pca=pca,
        cp=cp,
        fft_data=fft_data,
        peaks_info=peaks,
        preprocessed_path=preprocessed_path,
    )


def run_over_time_phase_based(
    config: PipelineConfig,
    preprocessed_dir: str,
    cache_dir: str,
    phase_config: Optional[PhaseConfig] = None,
    keep_full: bool = False,
    log: bool = True,
) -> OverTimeResult:
    """Análogo a run_over_time mas usa o pipeline phase-based.

    Default keep_full=False — só guarda CompactVideoResult com fft_data
    (necessário pra trend_analysis). Não preserva pca.score nem cp.unmixed
    (controle de memória; phase signals já moram em disco como cache).
    """
    phase_cfg = phase_config or PhaseConfig()
    os.makedirs(cache_dir, exist_ok=True)

    batches: dict[str, list[str]] = {
        "february": list(config.batches.february),
        "april": list(config.batches.april),
    }
    video_paths = batches["february"] + batches["april"]
    if not video_paths:
        raise ValueError("Nenhum vídeo definido em config.batches.")

    per_video: dict[str, CompactVideoResult | SingleVideoResult] = {}
    highest: dict[str, list[float]] = {}

    for i, video_path in enumerate(video_paths):
        basename = os.path.basename(video_path)
        pp = os.path.join(preprocessed_dir, basename)
        cache = os.path.join(cache_dir, basename.replace(".mp4", ".npz"))

        if log:
            cache_status = "cache hit" if os.path.exists(cache) else "computing"
            print(f"  [{i+1}/{len(video_paths)}] {basename} ({cache_status})", flush=True)

        result = analyse_phase_video(pp, cache, config, phase_cfg)
        highest[video_path] = [info.highest_freq for info in result.peaks_info.values()]

        if keep_full:
            per_video[video_path] = result
        else:
            per_video[video_path] = CompactVideoResult(
                video_info=result.video_info,
                peaks_info=result.peaks_info,
                fft_data=result.fft_data,
            )
            del result
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
