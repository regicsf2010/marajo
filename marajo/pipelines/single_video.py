"""Orquestrador do pipeline para um único vídeo (replica main_single_video.ipynb)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Union

from marajo.config import PipelineConfig
from marajo.decomposition.cp import CPResult, run_cp_on_components
from marajo.decomposition.pca import PCAResult, compute_pca
from marajo.io.roi import ROI
from marajo.io.video import VideoInfo, load_grayscale_dataset, video_status
from marajo.modal.fft import ComponentFFT, compute_fft_for_components
from marajo.modal.peaks import PeakInfo, highest_peak_frequencies
from marajo.preprocessing.video_prep import preprocess_video


@dataclass
class SingleVideoResult:
    video_info: VideoInfo
    pca: PCAResult
    cp: CPResult
    fft_data: dict[int, ComponentFFT]
    peaks_info: dict[int, PeakInfo]
    preprocessed_path: str


def analyse_preprocessed_video(
    preprocessed_path: str,
    config: PipelineConfig,
    full_pca: bool = False,
) -> SingleVideoResult:
    """Carrega um vídeo já pré-processado e roda PCA → CP → FFT → peaks.

    `full_pca=False` (default) cappa o PCA em `num_pcs` pra economizar memória —
    o `score` resultante fica (n_pixels, num_pcs) em vez de (n_pixels, n_frames).
    Use `full_pca=True` quando precisar dos autovalores completos (ex.: plot
    diagnóstico de variância acumulada).
    """
    info = video_status(preprocessed_path)
    dataset = load_grayscale_dataset(preprocessed_path)

    n_pca_comp = None if full_pca else config.decomposition.num_pcs
    pca = compute_pca(dataset, n_components=n_pca_comp)
    del dataset  # libera frames brutos antes de seguir
    cp = run_cp_on_components(pca, n_pc=config.decomposition.num_pcs, config=config.cp)

    fft_data = compute_fft_for_components(cp.unmixed, info.fps, range(config.decomposition.num_pcs))
    peaks = highest_peak_frequencies(fft_data, n_peaks=config.modal.n_peaks)

    return SingleVideoResult(
        video_info=info,
        pca=pca,
        cp=cp,
        fft_data=fft_data,
        peaks_info=peaks,
        preprocessed_path=preprocessed_path,
    )


def run_single_video(
    in_video_path: str,
    out_video_path: str,
    roi: Optional[Union[ROI, tuple[int, int, int, int]]],
    config: PipelineConfig,
    skip_preprocess_if_exists: bool = False,
) -> SingleVideoResult:
    """Pipeline completo: pré-processa o vídeo (se necessário) e analisa."""
    if not (skip_preprocess_if_exists and os.path.exists(out_video_path)):
        preprocess_video(
            in_path=in_video_path,
            out_path=out_video_path,
            roi=roi,
            config=config.preprocess.__class__(  # PreprocessConfig
                num_frames=config.preprocess.num_frames,
                fps=config.preprocess.fps,
                scale=config.preprocess.scale,
            ),
        )
    return analyse_preprocessed_video(out_video_path, config)
