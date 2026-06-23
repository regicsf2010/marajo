"""Anotação interativa de ROIs pra 3ª campanha (`3_camp/`).

Percorre `3_camp/` na ordem (dia → sessão _1/_2 → período m/n), encontra o .mp4
de cada combinação e abre a janela de desenho SÓ pros vídeos que ainda não têm
ROI em `rois/rois.json` (pula os já anotados). ENTER confirma, ESC pula o vídeo
atual, fechar a janela encerra.

Os basenames dos vídeos são únicos (timestamp), então convivem com as ROIs do
dataset antigo no mesmo `rois.json` sem colisão.

Uso (rodar de dentro de `marajo/`, com display disponível):
    python scripts/annotate_3camp_rois.py                # todos os 56 que faltam
    python scripts/annotate_3camp_rois.py --period n     # só noite
    python scripts/annotate_3camp_rois.py --period m     # só manhã
    python scripts/annotate_3camp_rois.py --plant 2      # só a planta estressada
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from marajo.io.roi import annotate_and_save_roi, load_rois

CAMP_ROOT = "/home/ygarasab/mestrado/tese/3_camp"
JSON_PATH = "rois/rois.json"


def find_videos(period: str | None, plant: str | None) -> list[tuple[str, str, str, str]]:
    """Devolve [(day, session, period, video_path)] ordenado."""
    items: list[tuple[str, str, str, str]] = []
    for day in sorted(os.listdir(CAMP_ROOT)):
        day_dir = os.path.join(CAMP_ROOT, day)
        if not (os.path.isdir(day_dir) and day.isdigit()):
            continue
        for session in sorted(os.listdir(day_dir)):  # 20260525_1, _2
            if plant and not session.endswith(f"_{plant}"):
                continue
            sess_dir = os.path.join(day_dir, session)
            if not os.path.isdir(sess_dir):
                continue
            for per in sorted(os.listdir(sess_dir)):  # m, n (ou 240/)
                if period and per != period:
                    continue
                per_dir = os.path.join(sess_dir, per)
                if not os.path.isdir(per_dir):
                    continue
                vids = sorted(f for f in os.listdir(per_dir) if f.endswith(".mp4"))
                for v in vids:
                    items.append((day, session, per, os.path.join(per_dir, v)))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", choices=["m", "n"], default=None)
    ap.add_argument("--plant", choices=["1", "2"], default=None)
    ap.add_argument("--overwrite", action="store_true", help="redesenhar mesmo se já houver ROI")
    args = ap.parse_args()

    videos = find_videos(args.period, args.plant)
    existing = load_rois(JSON_PATH)
    todo = [v for v in videos if args.overwrite or os.path.basename(v[3]) not in existing]

    print(f"{len(videos)} vídeos no filtro | {len(todo)} sem ROI (a desenhar)")
    if not todo:
        print("Nada a fazer. Tudo já anotado.")
        return

    for i, (day, session, per, path) in enumerate(todo, 1):
        tag = "ESTRESSADA" if session.endswith("_2") else "controle"
        print(f"[{i}/{len(todo)}] {day} {session}/{per} ({tag}) -> {os.path.basename(path)}")
        roi = annotate_and_save_roi(path, JSON_PATH, overwrite=args.overwrite)
        if roi is None:
            print("   (pulado / cancelado)")
        else:
            print(f"   ROI salva: {roi.to_list()}")


if __name__ == "__main__":
    main()
