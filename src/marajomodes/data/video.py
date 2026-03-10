import cv2 as cv
import tkinter as tk
import subprocess
import json

def get_screen_size():
    """
    Retorna a largura e a altura da tela em pixels.
    Usado internamente para centralizar janelas. Requer tkinter.
    """
    root = tk.Tk()
    root.withdraw()  # não mostra janela
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h

def roi_selection(video_path):
    """
    Abre o primeiro frame do vídeo em uma janela para o usuário selecionar
    a região de interesse (ROI) com o mouse. Arraste para desenhar um retângulo.
    Ao terminar, pressione ENTER ou ESC. As coordenadas (x, y, w, h) são
    impressas no terminal para uso em pre_processing(..., roi=(x, y, w, h)).
    """
    drawing = False
    ix, iy = -1, -1
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, frame, frame_copy

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv.EVENT_MOUSEMOVE:
            if drawing:
                frame = frame_copy.copy()
                cv.rectangle(frame, (ix, iy), (x, y), (0, 0, 255), 2)

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            w = abs(x - ix)
            h = abs(y - iy)
            print(f"x = {x0}, y = {y0}, w = {w}, h = {h} | {x0}, {y0}, {w}, {h}")
            cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

    # === Abrir vídeo ===
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Não foi possível ler o vídeo")
        
    frame_copy = frame.copy()
    
    win_name = "Selecione a ROI (ENTER ou ESC para sair)"

    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    h, w = frame.shape[:2]
    MAX_W, MAX_H = 800, 800

    scale = min(MAX_W / w, MAX_H / h, 1.0)
    win_w, win_h = int(w * scale), int(h * scale)

    cv.resizeWindow(win_name, win_w, win_h)

    screen_w, screen_h = get_screen_size()
    cv.moveWindow(win_name, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    
    cv.setMouseCallback(win_name, mouse_callback)

    while cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow(win_name, frame)
        if cv.waitKey(1) in (13, 27):
            break

    cv.destroyAllWindows()
    
def video_rotation(video_path):
    """
    Retorna o ângulo de rotação do vídeo em graus (0, 90, 180 ou 270).
    Usa ffprobe para ler metadados do arquivo. Se não houver informação
    de rotação, retorna 0.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-of", "json",
        video_path
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    data = json.loads(result.stdout)
    try:
        rotation = int(data["streams"][0]["side_data_list"][0]["rotation"])
    except (KeyError, IndexError):
        rotation = 0

    # Normalizar
    rotation = rotation % 360
    return rotation

def video_status(video_path):
    """
    Exibe no terminal informações do vídeo: caminho, FPS, largura/altura,
    quantidade de frames, duração, modo de cor, shape do primeiro frame
    e ângulo de rotação. Útil para inspeção rápida antes do processamento.
    """
    video = cv.VideoCapture(video_path)
    print(f'PATH = {video_path}')
    
    fps = round(video.get(cv.CAP_PROP_FPS))
    print(f'FPS = {fps}')
    
    print(f'WIDTH, HEIGHT = {int(video.get(cv.CAP_PROP_FRAME_WIDTH))}, {int(video.get(cv.CAP_PROP_FRAME_HEIGHT))}')
    
    num_frames = round(video.get(cv.CAP_PROP_FRAME_COUNT))
    print(f'FRAME COUNT = {num_frames}')
    
    duration = num_frames / fps
    print(f'DURATION (s) = {duration:.3}')   
    
    modes = {0: 'BGR', 1: 'RGB', 2: 'GRAY', 3: 'YUYV'}
    
    print(f'MODE = {modes[int(video.get(cv.CAP_PROP_MODE))]}')
    
    print(f'SHAPE = {video.read()[1].shape}')
    
    print(f'ROTATION = {video_rotation(video_path)}')


def pre_processing(in_video_path, out_video_path, num_frames, fps=None, scale=0.2, roi=None):
    """
    Pré-processa um vídeo e grava um novo arquivo.

    Lê até num_frames quadros do vídeo de entrada, converte para escala de cinza,
    opcionalmente recorta uma região (roi) e redimensiona. O vídeo de saída é
    gravado em out_video_path.

    Parâmetros
    ----------
    in_video_path : str
        Caminho do vídeo de entrada.
    out_video_path : str
        Caminho do vídeo de saída (será sobrescrito se existir).
    num_frames : int
        Número máximo de frames a processar.
    fps : float, opcional
        FPS do vídeo de saída. Se None, usa o FPS do vídeo de entrada.
    scale : float, opcional
        Fator de redimensionamento (0 a 1). Ex.: 0.2 reduz para 20% do tamanho.
    roi : tuple (x, y, w, h), opcional
        Região de interesse para recortar antes de redimensionar. Use roi_selection()
        para obter esses valores interativamente.

    Levanta
    -------
    IOError
        Se o vídeo de entrada não puder ser aberto.
    """
    video = cv.VideoCapture(in_video_path)

    if not video.isOpened():
        raise IOError("Erro ao abrir o vídeo")

    if fps is None:
        fps = video.get(cv.CAP_PROP_FPS)

    if roi is not None:
        x, y, w, h = roi
        base_w, base_h = w, h
    else:
        base_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        base_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    width = int(base_w * scale)
    height = int(base_h * scale)

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    out = cv.VideoWriter(out_video_path, fourcc, fps, (width, height), isColor=False)

    count = 0

    while count < num_frames:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if roi is not None:
            gray = gray[y : y + h, x : x + w]

        gray_small = cv.resize(gray, (width, height), interpolation=cv.INTER_AREA)
        out.write(gray_small)
        count += 1

    video.release()
    out.release()
    cv.destroyAllWindows()