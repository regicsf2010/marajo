import cv2 as cv
import tkinter as tk
import numpy as np
import os
import json


def get_screen_size():
    root = tk.Tk()
    root.withdraw()  # não mostra janela
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    root.destroy()
    return screen_w, screen_h

# def roi_selection(video_path):
#     drawing = False
#     ix, iy = -1, -1
    
#     def mouse_callback(event, x, y, flags, param):
#         nonlocal ix, iy, drawing, frame, frame_copy

#         if event == cv.EVENT_LBUTTONDOWN:
#             drawing = True
#             ix, iy = x, y

#         elif event == cv.EVENT_MOUSEMOVE:
#             if drawing:
#                 frame = frame_copy.copy()
#                 cv.rectangle(frame, (ix, iy), (x, y), (0, 0, 255), 2)

#         elif event == cv.EVENT_LBUTTONUP:
#             drawing = False
#             x0, y0 = min(ix, x), min(iy, y)
#             w = abs(x - ix)
#             h = abs(y - iy)
#             print(f"x = {x0}, y = {y0}, w = {w}, h = {h} | {x0}, {y0}, {w}, {h}")
#             cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

#     # === Abrir vídeo ===
#     cap = cv.VideoCapture(video_path)
#     ret, frame = cap.read()
#     cap.release()

#     if not ret:
#         raise RuntimeError("Não foi possível ler o vídeo")
        
#     frame_copy = frame.copy()
    
#     win_name = "Selecione a ROI (ENTER ou ESC para sair)"

#     cv.namedWindow(win_name, cv.WINDOW_NORMAL)

#     h, w = frame.shape[:2]
#     MAX_W, MAX_H = 800, 800

#     scale = min(MAX_W / w, MAX_H / h, 1.0)
#     win_w, win_h = int(w * scale), int(h * scale)

#     cv.resizeWindow(win_name, win_w, win_h)

#     screen_w, screen_h = get_screen_size()
#     cv.moveWindow(win_name, (screen_w - win_w) // 2, (screen_h - win_h) // 2)
    
#     cv.setMouseCallback(win_name, mouse_callback)

#     while cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) >= 1:
#         cv.imshow(win_name, frame)
#         if cv.waitKey(1) in (13, 27):
#             break

#     cv.destroyAllWindows()




def load_rois(json_path="rois/rois.json"):
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_rois(rois_data, json_path="rois/rois.json"):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rois_data, f, indent=2, ensure_ascii=False)


def get_first_frame(video_path):
    cap = cv.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Não foi possível ler o vídeo: {video_path}")

    return frame


def select_roi(video_path, max_w=800, max_h=800):
    drawing = False
    ix, iy = -1, -1
    selected_roi = None

    frame = get_first_frame(video_path)
    frame_copy = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, frame, frame_copy, selected_roi

        if event == cv.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv.EVENT_MOUSEMOVE and drawing:
            frame = frame_copy.copy()
            cv.rectangle(frame, (ix, iy), (x, y), (0, 0, 255), 2)

        elif event == cv.EVENT_LBUTTONUP:
            drawing = False
            x0, y0 = min(ix, x), min(iy, y)
            w = abs(x - ix)
            h = abs(y - iy)

            selected_roi = [int(x0), int(y0), int(w), int(h)]

            frame = frame_copy.copy()
            cv.rectangle(frame, (x0, y0), (x0 + w, y0 + h), (255, 0, 0), 2)

    win_name = "Selecione a ROI (ENTER confirma | ESC cancela)"

    cv.namedWindow(win_name, cv.WINDOW_NORMAL)

    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    win_w, win_h = int(w * scale), int(h * scale)

    cv.resizeWindow(win_name, win_w, win_h)

    screen_w, screen_h = get_screen_size()
    cv.moveWindow(win_name, (screen_w - win_w) // 2, (screen_h - win_h) // 2)

    cv.setMouseCallback(win_name, mouse_callback)

    confirmed = False

    while cv.getWindowProperty(win_name, cv.WND_PROP_VISIBLE) >= 1:
        cv.imshow(win_name, frame)
        key = cv.waitKey(1)

        if key == 13:  # ENTER
            confirmed = selected_roi is not None
            break
        elif key == 27:  # ESC
            break

    cv.destroyAllWindows()

    return selected_roi if confirmed else None


def annotate_and_save_roi(video_path, json_path="rois/rois.json", overwrite=False):
    rois_data = load_rois(json_path)
    video_name = os.path.basename(video_path)

    if video_name in rois_data and not overwrite:
        return rois_data[video_name]

    roi = select_roi(video_path)

    if roi is None:
        return None

    rois_data[video_name] = roi
    save_rois(rois_data, json_path)

    return roi


def get_roi_for_video(video_path, json_path="rois/rois.json"):
    rois_data = load_rois(json_path)
    video_name = os.path.basename(video_path)
    return rois_data.get(video_name)