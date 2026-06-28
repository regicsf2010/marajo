"""011.6 — Espectrogramas tempo-frequência por dia.

Pra cada dia (1º ângulo), computa espectrograma STFT do CP 0 (mais energético
do PCA) com janela 2s (60 FPS → 120 samples) e hop 0.5s. Saída: figura
mostrando energia × tempo × frequência. 16 figuras por frontend.

Foco visual: dá pra ver transientes? A energia é estacionária ao longo dos 10s?
A banda 0.5-2 Hz tem peaks que aparecem/somem ao longo do tempo?
"""

from __future__ import annotations

import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as scipy_signal

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.config import PipelineConfig
from marajo.pipelines.over_time import run_over_time
from marajo.pipelines.phase_based import run_over_time_phase_based
from marajo.preprocessing.phase_pyramid import PhaseConfig


CONFIG = "configs/011-60fps.yaml"
PREPROC_DIR = "out/all_angles_60fps/"
CACHE_DIR = "out/phase_cache_60fps/"
OUT_DIR = "out/experimentos/011-6/"


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> int:
    m = _DAY_RE.search(path)
    return int(m.group(1)) if m else 0


def plot_spectrogram_cp0(over_time_result, frontend: str, out_subdir: str):
    """Pra cada dia, pega 1º ângulo, computa STFT do CP 0, salva fig."""
    os.makedirs(out_subdir, exist_ok=True)
    seen_days = set()
    for path in over_time_result.video_order:
        day = day_from_path(path)
        if day in seen_days:
            continue
        seen_days.add(day)
        res = over_time_result.per_video[path]
        if res.fft_data is None or 0 not in res.fft_data:
            continue
        signal = res.fft_data[0].signal
        fps = res.video_info.fps

        # STFT: janela 2s, hop 0.5s
        win_samples = max(64, int(fps * 2.0))
        hop_samples = max(8, int(fps * 0.5))
        f, t, Sxx = scipy_signal.spectrogram(
            signal, fs=fps,
            nperseg=win_samples,
            noverlap=win_samples - hop_samples,
            scaling="density",
        )

        # Limit freq display to 0-10 Hz
        fmax_idx = np.searchsorted(f, 10.0)
        f_b = f[:fmax_idx]
        Sxx_b = Sxx[:fmax_idx]
        Sxx_db = 10 * np.log10(np.maximum(Sxx_b, 1e-20))

        fig, ax = plt.subplots(figsize=(10, 5))
        pcm = ax.pcolormesh(t, f_b, Sxx_db, shading="gouraud", cmap="viridis")
        ax.axhspan(0.5, 2.0, color="orange", alpha=0.18)
        ax.axhspan(2.0, 5.0, color="cyan", alpha=0.12)
        ax.set_xlabel("tempo (s)")
        ax.set_ylabel("frequência (Hz)")
        ax.set_title(f"Espectrograma — {frontend} CP 0 — dia {day} (60 FPS)")
        fig.colorbar(pcm, ax=ax, label="PSD (dB)")
        out_path = os.path.join(out_subdir, f"spec_{frontend}_{day}.png")
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PNG] {out_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    config = PipelineConfig.load(CONFIG)

    print("=== pixel ===")
    over_time_pixel = run_over_time(
        config=config,
        out_dir=PREPROC_DIR,
        do_preprocess=False,
        keep_fft_data=True,
    )
    plot_spectrogram_cp0(over_time_pixel, "pixel", os.path.join(OUT_DIR, "pixel"))
    del over_time_pixel
    import gc; gc.collect()

    print("\n=== phase ===")
    phase_cfg = PhaseConfig(
        n_scales=3, n_orientations=2, subsample_factor=4, use_phase_velocity=True,
    )
    over_time_phase = run_over_time_phase_based(
        config=config,
        preprocessed_dir=PREPROC_DIR,
        cache_dir=CACHE_DIR,
        phase_config=phase_cfg,
    )
    plot_spectrogram_cp0(over_time_phase, "phase", os.path.join(OUT_DIR, "phase"))


if __name__ == "__main__":
    main()
