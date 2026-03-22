import cv2 as cv
import tkinter as tk
import numpy as np

def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # não mostra janela
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h

def roi_selection(video_path):
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