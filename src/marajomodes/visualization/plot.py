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