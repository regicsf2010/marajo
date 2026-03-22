

    
def extract_signal(video_path):
    cap = cv.VideoCapture(video_path)

    ret, dataset = cap.read()
    if not ret:
        raise RuntimeError("Erro ao abrir vídeo")

    if dataset.ndim == 3:
        dataset = dataset[:, :, 0]
        
    dataset = dataset.astype(np.float32)

    signal = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame.ndim == 3:
            frame = frame[:, :, 0]
            
        frame = frame.astype(np.float32)

        # Diferença temporal
        diff = cv.absdiff(frame, prev)

        # Soma por coluna (movimento lateral)
        col_sum = np.sum(diff, axis=0)

        # Centroide no eixo X
        x_positions = np.arange(len(col_sum))

        if np.sum(col_sum) > 0:
            cx = np.sum(x_positions * col_sum) / np.sum(col_sum)
        else:
            cx = 0

        signal.append(cx)

        prev = frame

    cap.release()

    signal = np.array(signal)

    # Remove tendência DC
    signal -= np.mean(signal)

    return signal



def compute_fft(signal, fps):
    N = len(signal)

    # janela de Hanning (melhora muito o espectro)
    window = np.hanning(N)
    signal = signal * window # Multiplica o sinal por uma janela suave que começa e termina em zero. (Evita descontinuidade nas bordas → vazamento espectral (spectral leakage).)

    freqs = np.fft.rfftfreq(N, d = 1 / fps) # retorna só frequências positivas
    fft_vals = np.abs(np.fft.rfft(signal)) # aplica a fft, obtém valores complexos e depois a amplitude com o módulo

    return freqs, fft_vals