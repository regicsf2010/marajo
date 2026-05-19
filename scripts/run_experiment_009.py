"""Experimento 009 — Visualização tempo-frequência da migração de energia.

Gera 3 famílias de plots pra ver se nossa interpretação física ("energia
espectral migra pra freqs menores ao longo dos 8 dias de fevereiro") é
visualmente sustentável.

1. Waterfall PSD por dia (heatmap dia × freq, batch a batch).
2. Energia por sub-banda × dia (5 séries por batch).
3. Espectrogramas individuais (Day 1, 4, 8 × batch).

Roda em DOIS pipelines em paralelo: pixel-grayscale (out/all_angles/) e
phase-based (out/phase_cache/). Resultados salvos em out/experimentos/009/.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram

from marajo.config import PipelineConfig
from marajo.decomposition.cp import run_cp_on_components
from marajo.decomposition.pca import compute_pca
from marajo.io.video import load_grayscale_dataset, video_status
from marajo.modal.fft import compute_fft_for_components
from marajo.preprocessing.phase_pyramid import (
    PhaseConfig,
    compute_and_cache,
    flatten_signals,
)


_DAY_RE = re.compile(r"(\d{8})")


def day_from_path(path: str) -> int:
    m = _DAY_RE.search(path)
    return int(m.group(1)) if m else 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualização tempo-frequência (009).")
    p.add_argument("--config", default="configs/004-2-fine-bands.yaml")
    p.add_argument("--preprocessed-dir", default="out/all_angles/")
    p.add_argument("--phase-cache-dir", default="out/phase_cache/")
    p.add_argument("--plots-dir", default="out/experimentos/009/")
    p.add_argument("--freq-max-plot", type=float, default=30.0)
    return p.parse_args()


def compute_unmixed_per_video(
    config: PipelineConfig,
    video_paths: list[str],
    preprocessed_dir: str,
    phase_cache_dir: str | None = None,
    phase_config: PhaseConfig | None = None,
) -> dict[str, tuple[np.ndarray, int, int]]:
    """Pra cada vídeo, retorna (unmixed[n_frames, n_pcs], fps, day).

    Se phase_cache_dir é None → usa pipeline pixel-grayscale.
    Caso contrário → usa phase-based (cache de fase em disco).
    """
    n_pcs = config.decomposition.num_pcs
    out: dict[str, tuple[np.ndarray, int, int]] = {}

    for vp in video_paths:
        pp = os.path.join(preprocessed_dir, os.path.basename(vp))
        info = video_status(pp)
        day = day_from_path(vp)

        if phase_cache_dir is None:
            dataset = load_grayscale_dataset(pp)
        else:
            cache = os.path.join(phase_cache_dir, os.path.basename(vp).replace(".mp4", ".npz"))
            data = compute_and_cache(pp, cache, phase_config or PhaseConfig())
            dataset = flatten_signals(data)

        pca = compute_pca(dataset, n_components=n_pcs)
        del dataset
        cp = run_cp_on_components(pca, n_pc=n_pcs, config=config.cp)
        out[vp] = (cp.unmixed.copy(), int(info.fps), day)
        del pca, cp

    return out


def aggregate_psd_per_day(
    unmixed_data: dict[str, tuple[np.ndarray, int, int]],
    batches: dict[str, list[str]],
    freq_max_plot: float,
) -> dict[str, dict[int, np.ndarray]]:
    """Pra cada batch e dia, retorna PSD médio (sobre vídeos do dia e CPs)."""
    out: dict[str, dict[int, np.ndarray]] = {b: {} for b in batches}
    freqs_ref = None

    for batch_name, paths in batches.items():
        by_day: dict[int, list[np.ndarray]] = {}
        for vp in paths:
            unmixed, fps, day = unmixed_data[vp]
            # PSD agregado sobre os CPs do vídeo (média)
            psds = []
            for cp_idx in range(unmixed.shape[1]):
                signal = unmixed[:, cp_idx]
                n = len(signal)
                freqs = np.fft.rfftfreq(n, d=1.0 / fps)
                psd = np.abs(np.fft.rfft(signal)) ** 2
                mask = (freqs > 0) & (freqs <= freq_max_plot)
                if freqs_ref is None:
                    freqs_ref = freqs[mask]
                psds.append(psd[mask])
            psd_mean_video = np.mean(np.stack(psds, axis=0), axis=0)
            by_day.setdefault(day, []).append(psd_mean_video)

        for day, psds in by_day.items():
            out[batch_name][day] = np.mean(np.stack(psds, axis=0), axis=0)

    out["_freqs"] = freqs_ref  # type: ignore[assignment]
    return out


def energy_by_band_per_day(
    psd_by_day: dict[str, dict[int, np.ndarray]],
    freqs: np.ndarray,
    bands: list[tuple[float, float]],
) -> dict[str, dict[tuple[float, float], dict[int, float]]]:
    """Pra cada batch, banda, dia: energia total na banda."""
    out: dict[str, dict[tuple[float, float], dict[int, float]]] = {}
    for batch_name, by_day in psd_by_day.items():
        if batch_name.startswith("_"):
            continue
        out[batch_name] = {}
        for low, high in bands:
            mask = (freqs >= low) & (freqs < high)
            out[batch_name][(low, high)] = {
                day: float(np.sum(psd[mask])) for day, psd in by_day.items()
            }
    return out


def plot_waterfall(psd_data: dict, freqs: np.ndarray, frontend: str, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    vmin, vmax = float("inf"), float("-inf")
    matrices: dict = {}
    for batch_name in ("february", "april"):
        by_day = psd_data[batch_name]
        days_sorted = sorted(by_day.keys())
        mat = np.stack([by_day[d] for d in days_sorted], axis=0)
        # Normaliza por dia (cada linha soma 1) pra evidenciar mudança de FORMA
        row_sum = mat.sum(axis=1, keepdims=True)
        mat_norm = mat / np.where(row_sum > 0, row_sum, 1)
        # Log pra realçar baixa freq
        mat_log = np.log10(mat_norm + 1e-12)
        matrices[batch_name] = (mat_log, days_sorted)
        vmin = min(vmin, mat_log.min())
        vmax = max(vmax, mat_log.max())

    for ax, batch_name in zip(axes, ("february", "april")):
        mat_log, days_sorted = matrices[batch_name]
        im = ax.imshow(
            mat_log, aspect="auto", origin="lower",
            extent=[freqs[0], freqs[-1], 0.5, len(days_sorted) + 0.5],
            cmap="viridis", vmin=vmin, vmax=vmax,
        )
        ax.set_xlabel("Frequência (Hz)")
        ax.set_ylabel("Dia ordinal do batch")
        ax.set_title(f"{batch_name} — {frontend}", fontweight="bold")
        ax.set_yticks(range(1, len(days_sorted) + 1))
        ax.set_yticklabels([str(d)[-4:] for d in days_sorted], fontsize=8)
    fig.colorbar(im, ax=axes, label="log₁₀ (PSD normalizada por dia)",
                 location="right", shrink=0.8)
    fig.suptitle(
        f"Waterfall PSD por dia — frontend: {frontend}",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_energy_by_band(
    energy_data: dict, bands: list[tuple[float, float]], frontend: str, out_path: str
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True, sharey=True)
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(bands)))

    for ax, batch_name in zip(axes, ("february", "april")):
        for (low, high), color in zip(bands, colors):
            day_energy = energy_data[batch_name][(low, high)]
            days = sorted(day_energy.keys())
            xs = np.arange(1, len(days) + 1)
            ys = [day_energy[d] for d in days]
            ax.plot(xs, ys, marker="o", color=color, label=f"{low}-{high} Hz",
                    linewidth=2, markersize=7)
        ax.set_xlabel("Dia ordinal")
        ax.set_yscale("log")
        ax.set_title(f"{batch_name} — {frontend}", fontweight="bold")
        ax.set_xticks(range(1, len(days) + 1))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    axes[0].set_ylabel("Energia espectral (log)")
    fig.suptitle(
        f"Energia por sub-banda × dia — frontend: {frontend}",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def plot_individual_spectrograms(
    unmixed_data: dict,
    batches: dict[str, list[str]],
    days_to_show: list[int],
    frontend: str,
    out_path: str,
) -> None:
    """Espectrograma do CP 0 (arbitrário, fonte mais energética em geral)
    pra 1 vídeo de cada (batch, dia)."""
    fig, axes = plt.subplots(2, len(days_to_show), figsize=(5 * len(days_to_show), 8),
                              constrained_layout=True)

    for row, batch_name in enumerate(("february", "april")):
        # Mapeia day → primeiro vídeo do dia
        day_to_video: dict[int, str] = {}
        for vp in batches[batch_name]:
            d = day_from_path(vp)
            if d not in day_to_video:
                day_to_video[d] = vp

        all_days = sorted(day_to_video.keys())
        # Pra dias relativos (1, 4, 8) pega os índices correspondentes
        for col, day_rel in enumerate(days_to_show):
            ax = axes[row, col]
            if day_rel > len(all_days):
                ax.axis("off")
                continue
            d_actual = all_days[day_rel - 1]
            vp = day_to_video[d_actual]
            unmixed, fps, _ = unmixed_data[vp]
            signal = unmixed[:, 0]  # CP 0
            f, t, Sxx = spectrogram(
                signal, fs=fps, window="hann", nperseg=128, noverlap=96,
            )
            mask = f <= 30.0
            im = ax.pcolormesh(t, f[mask], np.log10(Sxx[mask] + 1e-12),
                               shading="gouraud", cmap="viridis")
            ax.set_title(f"{batch_name} · dia {day_rel} ({str(d_actual)[-4:]})", fontsize=10)
            ax.set_ylabel("Freq (Hz)")
            if row == 1:
                ax.set_xlabel("Tempo (s)")
    fig.suptitle(
        f"Espectrogramas individuais (CP 0) — frontend: {frontend}",
        fontsize=12, fontweight="bold",
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def save_raw_data(
    psd_data: dict, energy_data: dict, freqs: np.ndarray, frontend: str, out_dir: str
) -> None:
    # CSV: batch, day, freq, psd_normalized
    csv_path = os.path.join(out_dir, f"waterfall_{frontend}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["batch", "day", "freq_hz", "psd_normalized"])
        for batch_name in ("february", "april"):
            for day, psd in psd_data[batch_name].items():
                psd_n = psd / max(psd.sum(), 1e-12)
                for fr, p in zip(freqs, psd_n):
                    writer.writerow([batch_name, day, f"{fr:.3f}", f"{p:.6e}"])
    # JSON: estrutura completa
    json_path = os.path.join(out_dir, f"energy_by_band_{frontend}.json")
    payload = {
        "frontend": frontend,
        "bands_hz": list(energy_data["february"].keys()),
        "by_batch": {
            batch: {
                f"{low}-{high}": data for (low, high), data in batch_data.items()
            }
            for batch, batch_data in energy_data.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=lambda o: list(o) if hasattr(o, "__iter__") else float(o))
    print(f"[dados brutos] {csv_path}")
    print(f"[dados brutos] {json_path}")


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.config)
    os.makedirs(args.plots_dir, exist_ok=True)

    batches = {
        "february": list(config.batches.february),
        "april": list(config.batches.april),
    }
    video_paths = batches["february"] + batches["april"]
    bands = [(float(low), float(high)) for low, high in config.modal.bands]

    for frontend, phase_cache, phase_cfg in (
        ("pixel", None, None),
        ("phase", args.phase_cache_dir, PhaseConfig(n_scales=3, n_orientations=2, subsample_factor=4)),
    ):
        print(f"\n=== Frontend: {frontend} ===")
        unmixed_data = compute_unmixed_per_video(
            config, video_paths, args.preprocessed_dir, phase_cache, phase_cfg
        )
        print(f"Processados {len(unmixed_data)} vídeos.")

        psd_data = aggregate_psd_per_day(unmixed_data, batches, args.freq_max_plot)
        freqs = psd_data.pop("_freqs")

        energy_data = energy_by_band_per_day(psd_data, freqs, bands)

        plot_waterfall(
            psd_data, freqs, frontend,
            os.path.join(args.plots_dir, f"waterfall_{frontend}.png"),
        )
        plot_energy_by_band(
            energy_data, bands, frontend,
            os.path.join(args.plots_dir, f"energy_by_band_{frontend}.png"),
        )
        plot_individual_spectrograms(
            unmixed_data, batches, [1, 4, 8], frontend,
            os.path.join(args.plots_dir, f"spectrograms_{frontend}.png"),
        )

        save_raw_data(psd_data, energy_data, freqs, frontend, args.plots_dir)
        print(f"  plots em {args.plots_dir}/*_{frontend}.png")


if __name__ == "__main__":
    main()
