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