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
        
        

