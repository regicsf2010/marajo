import numpy as np
import matplotlib.pyplot as plt

def plot_signal(signal, save = False, w = 19, h = 8):
    
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
        plt.savefig('out/signal.png', bbox_inches = 'tight')
        
    
def plot_signals(signals, names, save = False, w = 19, h = 8):
    
    x = np.arange(len(signals[0])) + 1

    colors = ['k', 'r', 'b', 'g', 'c']
    plt.figure(figsize = (w, h))
    
    for i, signal in enumerate(signals):
        plt.plot(x, signal, linestyle = '-', color = colors[i], linewidth = 2, label = names[i])

    plt.title('Centroid of x position', fontsize = 22)
    plt.xlabel('frame', fontsize = 22)
    plt.ylabel('cx', fontsize = 22)
    plt.legend(loc = 'upper left', fancybox = True, shadow = True, fontsize = 20)

    # plt.xticks(x)
    plt.tick_params(axis = 'both', labelsize = 22)

    plt.tight_layout()
    
    if save:
        plt.savefig('out/signal.png', bbox_inches = 'tight')
        
        
def plot_freq(freqs, fft_vals, save = False, w = 19, h = 8):
    
    plt.figure(figsize = (12, 8))
    plt.plot(freqs, fft_vals, linestyle = '-', color = 'k', linewidth = 2, label = 'signal')

    plt.title('Amplitudes of frequencies', fontsize = 22)
    plt.xlabel('frequency (Hz)', fontsize = 22)
    plt.ylabel('amplitude', fontsize = 22)
    plt.legend(loc = 'upper right', fancybox = True, shadow = True, fontsize = 20)

    # plt.xticks(x)
    plt.tick_params(axis = 'both', labelsize = 22)

    plt.tight_layout()
    
    if save:
        plt.savefig('out/frequency.png', bbox_inches = 'tight')
        
def plot_freqs(freqs, fft_vals, names, save = False, w = 19, h = 8):
    
    plt.figure(figsize = (w, h))
    
    colors = ['k', 'r', 'b', 'g', 'c']
    for i, (x, y) in enumerate(zip(freqs, fft_vals)):
        plt.plot(x, y, linestyle = '-', color = colors[i], linewidth = 2, label = names[i])

    plt.title('Amplitudes of frequencies', fontsize = 22)
    plt.xlabel('frequency (Hz)', fontsize = 22)
    plt.ylabel('amplitude', fontsize = 22)
    plt.legend(loc = 'upper right', fancybox = True, shadow = True, fontsize = 20)

    # plt.xticks(x)
    plt.tick_params(axis = 'both', labelsize = 22)

    plt.tight_layout()
    
    if save:
        plt.savefig('out/frequency.png', bbox_inches = 'tight')
        
        

def plot_pca_eigenvalues_with_threshold(V, x, w = 15, h = 8):
    V = np.asarray(V, dtype=np.float64).ravel()

    if V.size == 0:
        raise ValueError("V está vazio.")

    if np.any(V < 0):
        raise ValueError("Os autovalores em V devem ser não negativos.")

    x_input = x
    if x > 1:
        x = x / 100.0

    if not (0 < x <= 1):
        raise ValueError("x deve estar em (0,1] ou em (0,100].")

    explained_ratio = V / np.sum(V)
    cumulative_ratio = np.cumsum(explained_ratio)

    n_components = np.searchsorted(cumulative_ratio, x) + 1
    pc_idx = n_components - 1

    components = np.arange(1, len(V) + 1)

    fig, ax1 = plt.subplots(figsize=(w, h))

    # Autovalores
    ax1.plot(components, V, marker = 'o', linewidth=2, label='Autovalores')
    ax1.axvline(n_components, linestyle='--', linewidth=1.2, label=f'PC = {n_components}')
    ax1.plot(components[pc_idx], V[pc_idx], 'ro')

    ax1.set_xlabel('Componente principal', fontsize = 20)
    ax1.set_ylabel('Autovalor', fontsize = 20)
    ax1.set_title('Autovalores do PCA e variância acumulada', fontsize = 20)
    ax1.grid(True, alpha=0.3)

    plt.tick_params(axis = 'both', labelsize = 22)
    
    # Variância acumulada no segundo eixo
    ax2 = ax1.twinx()
    ax2.plot(components, cumulative_ratio, marker = 's', color = 'r', linewidth=2, label='Variância acumulada')
    ax2.axhline(x, linestyle='--', linewidth=1.2, label=f'Limiar = {x:.2%}')
    ax2.plot(components[pc_idx], cumulative_ratio[pc_idx], 'bs')
    ax2.set_ylabel('Variância acumulada', fontsize = 20)

    # Texto de marcação
    ax2.text(
        n_components,
        cumulative_ratio[pc_idx],
        f'  PC {n_components}\n  acum = {cumulative_ratio[pc_idx]:.2%}',
        va='bottom',
        ha='right',
        fontsize=10
    )

    # Legenda combinada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fancybox = True, shadow = True, fontsize = 20)
    
    plt.tick_params(axis = 'both', labelsize = 22)

    plt.tight_layout()
    
    return n_components


def plot_source_psd_phase(fft_data, peaks_info, video_features, nPC, w=20, h=15, save=False):

    t = np.arange(video_features["frames"]) / video_features["fps"]
    t_plot = np.asarray(t).ravel()

    fig, axes = plt.subplots(len(nPC), 3, figsize=(w, h), constrained_layout=True)

    if len(nPC) == 1:
        axes = np.expand_dims(axes, axis=0)

    for cax, comp_id in enumerate(nPC):
        item = fft_data[comp_id]

        signal = np.asarray(item["signal"]).ravel()
        freq_plot = np.asarray(item["f"]).ravel()
        fft_vals = np.asarray(item["v"]).ravel()

        psd = np.abs(fft_vals) ** 2
        phase = np.angle(fft_vals) ** 2

        mask = freq_plot > 0
        freq_pos = freq_plot[mask]
        psd_pos = psd[mask]
        phase_pos = phase[mask]

        top_freqs = np.asarray(peaks_info[comp_id]["top_freqs"]).ravel()
        top_amps = np.asarray(peaks_info[comp_id]["top_amps"]).ravel()

        peak_lines = [
            f"freq: {f:.2f} Hz | amp: {a:.2f}"
            for f, a in zip(top_freqs, top_amps)
        ]
        peak_text = "\n".join(peak_lines)

        # --- Source ---
        ax = axes[cax, 0]
        ax.plot(t_plot, signal, linewidth=1.5)
        ax.set_title(f"Source {comp_id}", fontsize=10)
        ax.tick_params(labelsize=9)

        # --- PSD ---
        ax = axes[cax, 1]
        ax.plot(freq_pos, psd_pos, linewidth=1.5, color="k")
        ax.plot(top_freqs, top_amps, "ro", markersize=4)

        ax.text(
            0.5, 0.5, peak_text,
            transform=ax.transAxes,
            fontsize=9,
            ha="left",
            va="center",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        ax.set_title(f"PSD {comp_id}", fontsize=10)
        ax.tick_params(labelsize=9)

        # --- Phase ---
        ax = axes[cax, 2]
        ax.plot(freq_pos, phase_pos, linewidth=1.5, color="k")
        ax.set_title(f"Phase {comp_id}", fontsize=10)
        ax.tick_params(labelsize=9)

    axes[-1, 0].set_xlabel("Time (s)")
    axes[-1, 1].set_xlabel("Frequency (Hz)")
    axes[-1, 2].set_xlabel("Frequency (Hz)")

    if save:
        fig.savefig("out/sfp.pdf", bbox_inches="tight")