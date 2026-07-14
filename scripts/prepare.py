import cv2 as cv


def get_output_video_specs(
    video: cv.VideoCapture,
    fps_out: int | None = None,
    scale_out: float = 0.2,
    roi: tuple[int, int, int, int] | None = None,
) -> tuple[int, int, float]:
    """
    Calcula largura, altura e FPS do vídeo de saída.
    """

    if scale_out <= 0:
        raise ValueError("scale_out deve ser maior que zero.")

    fps = fps_out if fps_out is not None else video.get(cv.CAP_PROP_FPS)

    if fps <= 0:
        raise ValueError("FPS inválido. Informe fps_out manualmente.")

    if roi is not None:
        x, y, w, h = roi

        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError("ROI inválida. Use o formato (x, y, w, h) com valores positivos.")

        base_w, base_h = w, h
    else:
        base_w = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        base_h = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    width = int(base_w * scale_out)
    height = int(base_h * scale_out)

    if width <= 0 or height <= 0:
        raise ValueError("Dimensões finais inválidas. Verifique scale_out e ROI.")

    return width, height, fps


def create_video_writer(
    out_video_path: str,
    width: int,
    height: int,
    fps: float,
) -> cv.VideoWriter:
    """
    Cria o objeto de escrita do vídeo de saída.
    """

    fourcc = cv.VideoWriter_fourcc(*"mp4v")

    return cv.VideoWriter(
        out_video_path,
        fourcc,
        fps,
        (width, height),
        isColor=False,
    )


def pre_processing(
    in_video_path: str,
    out_video_path: str,
    nframes_out: int,
    fps_out: int | None = None,
    scale_out: float = 0.2,
    roi: tuple[int, int, int, int] | None = None,
) -> None:
    """
    Realiza o pré-processamento do vídeo:
    - leitura dos frames;
    - conversão para escala de cinza;
    - recorte por ROI, se informado;
    - redimensionamento;
    - escrita do vídeo final.
    """

    if nframes_out <= 0:
        raise ValueError("nframes_out deve ser maior que zero.")

    video = cv.VideoCapture(in_video_path)

    if not video.isOpened():
        raise IOError(f"Erro ao abrir o vídeo: {in_video_path}")

    out = None

    try:
        width, height, fps = get_output_video_specs(
            video=video,
            fps_out=fps_out,
            scale_out=scale_out,
            roi=roi,
        )

        out = create_video_writer(
            out_video_path=out_video_path,
            width=width,
            height=height,
            fps=fps,
        )

        if not out.isOpened():
            raise IOError(f"Erro ao criar o vídeo de saída: {out_video_path}")

        frame_count = 0

        while frame_count < nframes_out:
            ret, frame = video.read()

            if not ret:
                break

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            if roi is not None:
                x, y, w, h = roi
                gray = gray[y : y + h, x : x + w]

            gray_small = cv.resize(
                gray,
                (width, height),
                interpolation=cv.INTER_AREA,
            )

            out.write(gray_small)

            frame_count += 1

    finally:
        video.release()

        if out is not None:
            out.release()

        cv.destroyAllWindows()