import numpy as np
import matplotlib.pyplot as plt

def plot_signal(signal, save=False, w=12, h=8, prefix='out'):
    """
    Plota o sinal temporal (ex.: posição do centróide por frame).

    Cria uma figura com o sinal no eixo Y e o número do frame no eixo X.
    Opcionalmente salva a figura em arquivo PNG.

    Parâmetros
    ----------
    signal : array-like
        Sinal 1D a ser plotado.
    save : bool, opcional
        Se True, salva a figura em {prefix}/signal.png (padrão False).
    w, h : float, opcional
        Largura e altura da figura em polegadas (padrão 12 e 8).
    prefix : str, opcional
        Pasta onde salvar a figura quando save=True (padrão 'out').
    """
    x = np.arange(len(signal)) + 1

    plt.figure(figsize = (w, h))
    plt.plot(x, signal, linestyle = '-', color = 'k', linewidth = 2, label = 'signal')

    plt.title('Centroid of x position', fontsize = 22)
    plt.xlabel('frame', fontsize = 22)
    plt.ylabel('cx', fontsize = 22)
    plt.legend(loc = 'upper left', fancybox = True, shadow = True, fontsize = 20)

    # plt.xticks(x)
    plt.tick_params(axis = 'both', labelsize = 22)

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{prefix}/signal.png', bbox_inches = 'tight')
        
        
        
def plot_freq(freqs, fft_vals, save=False, w=12, h=8, prefix='out'):
    """
    Plota o espectro de frequências (amplitude x frequência em Hz).

    Útil para visualizar o resultado de compute_fft e identificar picos.
    Opcionalmente salva a figura em arquivo PNG.

    Parâmetros
    ----------
    freqs : array-like
        Array de frequências em Hz.
    fft_vals : array-like
        Array de amplitudes do espectro.
    save : bool, opcional
        Se True, salva a figura em {prefix}/frequency.png (padrão False).
    w, h : float, opcional
        Largura e altura da figura em polegadas (padrão 12 e 8).
    prefix : str, opcional
        Pasta onde salvar a figura quando save=True (padrão 'out').
    """
    plt.figure(figsize=(w, h))
    plt.plot(freqs, fft_vals, linestyle = '-', color = 'k', linewidth = 2, label = 'signal')

    plt.title('Amplitudes of frequencies', fontsize = 22)
    plt.xlabel('frequência (Hz)', fontsize = 22)
    plt.ylabel('amplitude', fontsize = 22)
    plt.legend(loc = 'upper right', fancybox = True, shadow = True, fontsize = 20)

    # plt.xticks(x)
    plt.tick_params(axis = 'both', labelsize = 22)

    plt.tight_layout()
    
    if save:
        plt.savefig(f'{prefix}/frequency.png', bbox_inches = 'tight')


def plot_sources(t, freq, unmixed, n_show=None, fs=1.0):
    nFrames = len(t)
    half = round(nFrames / 2)
    n = n_show or unmixed.shape[1]

    fig, axs = plt.subplots(n, 3, figsize=(10, 2 * n))

    for i in range(n):
        axs[i, 0].plot(t, unmixed[:, i], "k", lw=1.5)
        axs[i, 0].set_title(f"Source {i+1}")
        axs[i, 0].set_xlabel("Time (s)")

        fft_mag = np.abs(np.fft.fft(unmixed[:, i]))**2
        axs[i, 1].plot(freq[1:half], fft_mag[1:half], "k", lw=1.5)
        axs[i, 1].set_title("PSD" if i == 0 else "")
        axs[i, 1].set_xlabel("Frequency (Hz)")

        fft_phase = np.angle(np.fft.fft(unmixed[:, i]))**2
        axs[i, 2].plot(freq[1:half], fft_phase[1:half], "k", lw=1.5)
        axs[i, 2].set_title("Phase" if i == 0 else "")
        axs[i, 2].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    return fig


def compute_fft_features_cuda(unmixed: np.ndarray, *, use_cuda: bool | None = None):
    """
    Compute FFT-derived features for all sources at once.

    Returns positive-frequency bins excluding DC, matching ``freq[1:half]`` where
    ``half = round(n_frames / 2)`` (same convention as ``plot_sources_large``).

    Parameters
    ----------
    unmixed
        Array with shape ``(n_frames, n_sources)``.
    use_cuda
        If True, use CuPy on the GPU (requires CuPy + CUDA).
        If False, use NumPy on CPU.
        If None, use CUDA when available, otherwise CPU.
    """
    unmixed = np.asarray(unmixed)
    n_frames = unmixed.shape[0]
    half = round(n_frames / 2)
    if unmixed.ndim != 2:
        raise ValueError(f"`unmixed` must be 2D (n_frames, n_sources). Got shape {unmixed.shape}.")

    if use_cuda is None:
        try:
            import cupy as cp

            use_cuda = bool(cp.cuda.is_available())
        except ImportError:
            use_cuda = False

    if use_cuda:
        import cupy as cp

        X = cp.asarray(unmixed)
        Xf = cp.fft.fft(X, axis=0)
        psd_full = cp.abs(Xf) ** 2
        phase_full = cp.angle(Xf) ** 2  # keeping MATLAB-like behavior

        # Exclude DC: use [1:half]
        psd_pos = cp.asnumpy(psd_full[1:half, :])
        phase_pos = cp.asnumpy(phase_full[1:half, :])
        return psd_pos, phase_pos

    # CPU fallback
    Xf = np.fft.fft(unmixed, axis=0)
    psd_pos = np.abs(Xf[1:half, :]) ** 2
    phase_pos = np.angle(Xf[1:half, :]) ** 2
    return psd_pos, phase_pos


def plot_sources_large(
    t,
    freq,
    unmixed,
    n_show=None,
    *,
    peak_count: int = 5,
    freq_max: float | None = None,
    psd_log: bool = False,
    row_height: float = 1.7,
    spectra: tuple[np.ndarray, np.ndarray] | None = None,
    dpi: int = 100,
):
    """
    Larger version of ``plot_sources`` with an extra column for top PSD peaks.

    Columns:
      1) time trace
      2) PSD (magnitude^2 of FFT)
      3) phase (kept MATLAB-like: angle(FFT)^2)
      4) top ``peak_count`` peaks in the PSD (frequency vs amplitude)

    Parameters
    ----------
    freq_max
        If set, zooms PSD/phase/peaks columns up to this frequency (Hz).
    psd_log
        If True, plot PSD amplitude on a log scale (helps visibility when peaks are
        much smaller than the noise floor).
    dpi
        Figure DPI (lower -> smaller file, less detail).
    """
    nFrames = len(t)
    half = round(nFrames / 2)
    n = n_show or unmixed.shape[1]

    # Only use positive-frequency bins excluding DC.
    f = freq[1:half]

    if freq_max is not None:
        max_idx = int(np.searchsorted(f, freq_max, side="right"))
        f = f[:max_idx]

    # Make a smaller (lighter) figure.
    # Do not share x so tick labels are shown on every row.
    fig, axs = plt.subplots(
        n,
        4,
        figsize=(16, max(4, row_height * n)),
        dpi=dpi,
    )
    axs = np.asarray(axs)
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)

    fig.subplots_adjust(hspace=0.45, wspace=0.25)

    if spectra is not None:
        psd_all, phase_all = spectra
        psd_all = np.asarray(psd_all)
        phase_all = np.asarray(phase_all)
        expected_len = len(freq[1:half])
        if psd_all.shape[0] != expected_len or phase_all.shape[0] != expected_len:
            raise ValueError(
                "Provided `spectra` does not match the expected positive-frequency "
                f"length. Expected {expected_len} bins, got psd={psd_all.shape[0]}, phase={phase_all.shape[0]}."
            )
    else:
        psd_all = None
        phase_all = None

    for i in range(n):
        signal = unmixed[:, i]
        axs[i, 0].plot(t, signal, "k", lw=1.8)
        axs[i, 0].set_title(f"Source {i+1}", fontsize=12)
        axs[i, 0].set_xlabel("Time (s)", fontsize=10)
        axs[i, 0].tick_params(axis="x", labelsize=10, labelbottom=True, bottom=True)
        axs[i, 0].grid(alpha=0.25)
        # A few fixed ticks so numbers are readable after zooming.
        time_ticks = np.linspace(float(t[0]), float(t[-1]), 4)
        axs[i, 0].set_xticks(time_ticks)
        axs[i, 0].set_xticklabels([f"{x:.2g}" for x in time_ticks], fontsize=10)

        if psd_all is not None and phase_all is not None:
            psd = psd_all[:, i]
            phase = phase_all[:, i]
        else:
            # Compute FFT per source (fallback).
            fft_vals = np.fft.fft(signal)
            psd_full = np.abs(fft_vals) ** 2
            psd = psd_full[1:half]
            phase_full = np.angle(fft_vals) ** 2  # keeping MATLAB behavior
            phase = phase_full[1:half]

        if freq_max is not None:
            psd = psd[: len(f)]
            phase = phase[: len(f)]

        # --- PSD ---
        axs[i, 1].plot(f, psd, "k", lw=1.5)
        if i == 0:
            axs[i, 1].set_title("PSD", fontsize=12)
        axs[i, 1].set_ylabel("Amplitude", fontsize=10)
        axs[i, 1].set_xlabel("Frequency (Hz)", fontsize=10)
        axs[i, 1].tick_params(axis="x", labelsize=10, labelbottom=True, bottom=True)
        axs[i, 1].grid(alpha=0.25)
        if psd_log:
            axs[i, 1].set_yscale("log")

        # --- Phase ---
        axs[i, 2].plot(f, phase, "k", lw=1.3)
        if i == 0:
            axs[i, 2].set_title("Phase", fontsize=12)
        axs[i, 2].set_xlabel("Frequency (Hz)", fontsize=10)
        axs[i, 2].tick_params(axis="x", labelsize=10, labelbottom=True, bottom=True)
        axs[i, 2].grid(alpha=0.25)

        # --- Top peaks (freq + amplitude) ---
        if len(psd) == 0:
            peak_freqs = np.array([])
            peak_amps = np.array([])
        else:
            k = min(peak_count, len(psd))
            top_idx_unsorted = np.argsort(psd)[-k:]
            # Rank from highest amplitude to lowest amplitude.
            top_idx = top_idx_unsorted[np.argsort(psd[top_idx_unsorted])[::-1]]
            peak_freqs = f[top_idx]
            peak_amps = psd[top_idx]

        # Table-like display (frequency + amplitude).
        axs[i, 3].axis("off")
        if i == 0:
            axs[i, 3].set_title(f"Top {peak_count} peaks", fontsize=12)

        # Format values to be readable at low frequencies.
        header = f"{'k':>2s}  {'freq(Hz)':>10s}  {'amp(PSD)':>12s}"
        lines = [header, "-" * len(header)]
        # Peak rank starts at 1.
        for j, (pf, pa) in enumerate(zip(peak_freqs, peak_amps), start=1):
            lines.append(f"{j:>2d}  {pf:>10.4g}  {pa:>12.3e}")
        table_txt = "\n".join(lines)
        axs[i, 3].text(
            0.02,
            0.98,
            table_txt,
            transform=axs[i, 3].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

        # Tighten x-limits (especially helpful when freq_max is set).
        if freq_max is not None:
            axs[i, 1].set_xlim(0, freq_max)
            axs[i, 2].set_xlim(0, freq_max)
            # x-limits for the table column are irrelevant (axis off).

            freq_ticks = np.linspace(0, float(freq_max), 5)
            axs[i, 1].set_xticks(freq_ticks)
            axs[i, 2].set_xticks(freq_ticks)
            tick_labels = [f"{x:.2g}" for x in freq_ticks]
            axs[i, 1].set_xticklabels(tick_labels, fontsize=10)
            axs[i, 2].set_xticklabels(tick_labels, fontsize=10)
        else:
            # Default ticks up to the max shown frequency.
            freq_ticks = np.linspace(float(f[0]), float(f[-1]), 5) if len(f) else []
            if len(freq_ticks):
                axs[i, 1].set_xticks(freq_ticks)
                axs[i, 2].set_xticks(freq_ticks)
                tick_labels = [f"{x:.2g}" for x in freq_ticks]
                axs[i, 1].set_xticklabels(tick_labels, fontsize=10)
                axs[i, 2].set_xticklabels(tick_labels, fontsize=10)

    plt.tight_layout(pad=1.6)
    return fig


def plot_source_simple(
    t,
    freq,
    unmixed,
    *,
    spectra: tuple[np.ndarray, np.ndarray] | None = None,
    n_show: int | None = None,
    peak_count: int = 10,
    freq_max: float | None = None,
    psd_log: bool = False,
    dpi: int = 90,
    row_height: float = 1.8,
):
    """
    Compact visualization: per-source PSD + a table of top PSD peaks.

    Parameters
    ----------
    spectra
        Output of :func:`compute_fft_features_cuda`, i.e. ``(psd_pos, phase_pos)``.
        Only ``psd_pos`` is used here.
    peak_count
        How many peaks to show (ranked by highest PSD amplitude).
    freq_max
        If set, zoom PSD and top-peak search to frequencies <= ``freq_max`` (Hz).
    psd_log
        If True, plot PSD on log scale.
    """
    nFrames = len(t)
    half = round(nFrames / 2)
    n = n_show or unmixed.shape[1]

    f_all = freq[1:half]
    if freq_max is not None:
        mask = f_all <= freq_max
        f = f_all[mask]
    else:
        f = f_all

    if spectra is not None:
        psd_pos, _ = spectra
        psd_pos = np.asarray(psd_pos)
        expected_len = len(f_all)
        if psd_pos.shape[0] != expected_len:
            raise ValueError(
                "Provided `spectra[0]` length does not match expected positive-frequency bins. "
                f"Expected {expected_len}, got {psd_pos.shape[0]}."
            )
        if freq_max is not None:
            psd_all = psd_pos[mask, :]
        else:
            psd_all = psd_pos
    else:
        # Fallback: compute PSD for shown sources only.
        psd_all = None

    fig, axs = plt.subplots(
        n,
        2,
        figsize=(14, max(4, row_height * n)),
        dpi=dpi,
    )
    axs = np.asarray(axs)
    if axs.ndim == 1:
        axs = axs.reshape(1, -1)

    fig.subplots_adjust(hspace=0.35, wspace=0.25)

    for i in range(n):
        if psd_all is None:
            signal = unmixed[:, i]
            fft_vals = np.fft.fft(signal)
            psd_full = np.abs(fft_vals) ** 2
            psd = psd_full[1:half]
            if freq_max is not None:
                psd = psd[mask]
        else:
            psd = psd_all[:, i]

        # PSD plot
        axs[i, 0].plot(f, psd, "k", lw=1.5)
        if i == 0:
            axs[i, 0].set_title("PSD", fontsize=12)
        axs[i, 0].set_xlabel("Frequency (Hz)", fontsize=10)
        axs[i, 0].set_ylabel("Amplitude", fontsize=10)
        axs[i, 0].tick_params(axis="x", labelsize=9, labelbottom=True, bottom=True)
        axs[i, 0].grid(alpha=0.25)
        if psd_log:
            axs[i, 0].set_yscale("log")

        # Table of top peaks
        axs[i, 1].axis("off")
        if i == 0:
            axs[i, 1].set_title(f"Top {peak_count} peaks", fontsize=12)

        k = min(peak_count, len(psd))
        if k == 0:
            peak_freqs = np.array([])
            peak_amps = np.array([])
        else:
            top_idx_unsorted = np.argsort(psd)[-k:]
            # Rank from highest amplitude to lowest amplitude.
            top_idx = top_idx_unsorted[np.argsort(psd[top_idx_unsorted])[::-1]]
            peak_freqs = f[top_idx]
            peak_amps = psd[top_idx]

        header = f"{'k':>2s}  {'freq(Hz)':>10s}  {'amp(PSD)':>12s}"
        lines = [header, "-" * len(header)]
        for j, (pf, pa) in enumerate(zip(peak_freqs, peak_amps), start=1):
            lines.append(f"{j:>2d}  {pf:>10.4g}  {pa:>12.3e}")
        table_txt = "\n".join(lines)
        axs[i, 1].text(
            0.02,
            0.98,
            table_txt,
            transform=axs[i, 1].transAxes,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
        )

        if freq_max is not None and len(f) > 0:
            tick_vals = np.linspace(0, float(freq_max), 5)
            axs[i, 0].set_xticks(tick_vals)
            axs[i, 0].set_xticklabels([f"{x:.2g}" for x in tick_vals], fontsize=9)

    plt.tight_layout(pad=1.4)
    return fig


def write_sources_top_peaks_csv(
    t,
    freq,
    spectra: tuple[np.ndarray, np.ndarray],
    csv_path: str,
    *,
    top_k: int = 10,
    freq_max: float | None = None,
):
    """
    Write a CSV of the top PSD peaks for each source.

    CSV columns: source_idx (0-based), rank (1..top_k), freq_hz, amp_psd
    """
    import csv

    nFrames = len(t)
    half = round(nFrames / 2)
    f_all = freq[1:half]

    psd_pos, _ = spectra
    psd_pos = np.asarray(psd_pos)

    expected_len = len(f_all)
    if psd_pos.shape[0] != expected_len:
        raise ValueError(
            "Provided `spectra[0]` length does not match expected positive-frequency bins. "
            f"Expected {expected_len}, got {psd_pos.shape[0]}."
        )

    if freq_max is not None:
        mask = f_all <= freq_max
        f = f_all[mask]
        psd_search = psd_pos[mask, :]
    else:
        f = f_all
        psd_search = psd_pos

    n_sources = psd_search.shape[1]
    top_k = int(top_k)

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["source_idx", "rank", "freq_hz", "amp_psd"])

        for src_idx in range(n_sources):
            psd = psd_search[:, src_idx]
            k = min(top_k, len(psd))
            if k == 0:
                continue

            top_idx_unsorted = np.argsort(psd)[-k:]
            top_idx = top_idx_unsorted[np.argsort(psd[top_idx_unsorted])[::-1]]
            peak_freqs = f[top_idx]
            peak_amps = psd[top_idx]

            for rank, (pf, pa) in enumerate(zip(peak_freqs, peak_amps), start=1):
                writer.writerow([src_idx, rank, float(pf), float(pa)])

    return csv_path


def plot_mode_shapes(mode_shapes, srcs, width, height):

    n_srcs = len(srcs)

    fig, axes = plt.subplots(2, n_srcs//2, figsize=(8,5.5))

    axes = axes.flatten()

    vmax = np.max(np.abs(mode_shapes))
    vmin = -vmax

    for i in range(n_srcs):

        S = mode_shapes[:, i].reshape(height, width)

        im = axes[i].imshow(
            S,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            origin="lower"
        )

        axes[i].set_title(f"Mode Shape {i+1}", fontsize=12)
        axes[i].axis("off")

        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig

def plot_modal_coord(modal_coord, t, freq, numSrc, nFrames):
    fig, axes = plt.subplots(numSrc, 3, figsize=(10, 2.5*numSrc), sharex='col')

    half = nFrames // 2

    for i in range(numSrc):

        signal = modal_coord[:, i]

        fft_vals = np.fft.fft(signal)
        psd = np.abs(fft_vals)**2
        phase = np.angle(fft_vals)**2   # keeping MATLAB behavior

        # --- Time coordinate ---
        axes[i, 0].plot(t, signal, lw=1.5)
        axes[i, 0].set_title(f"Coordinate {i+1}")
        axes[i, 0].tick_params(labelsize=9)
        axes[i, 0].grid(alpha=0.3)

        # --- PSD ---
        axes[i, 1].plot(freq[1:half], psd[1:half], lw=1.5, color='k')
        if i == 0:
            axes[i, 1].set_title("PSD")
        axes[i, 1].tick_params(labelsize=9)
        axes[i, 1].grid(alpha=0.3)

        # --- Phase ---
        axes[i, 2].plot(freq[1:half], phase[1:half], lw=1.5, color='k')
        if i == 0:
            axes[i, 2].set_title("Phase")
        axes[i, 2].tick_params(labelsize=9)
        axes[i, 2].grid(alpha=0.3)

    # axis labels on bottom row
    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[-1, 2].set_xlabel("Frequency (Hz)")

    plt.tight_layout()
    return fig