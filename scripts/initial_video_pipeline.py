import sys, argparse, os
import numpy as np

sys.path.insert(0, "src")

from moises.data import Video, load_video_dataset
import moises

def process_video(video_path, max_frames, scale=1, out_dir = "prep"):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Processing video from {video_path}")
    print("Scale", scale)
    print("Max Frames", max_frames)

    video = Video(video_path, max_frames=max_frames, scale=scale)
    video.load()

    dataset, mean = load_video_dataset(video)

    print("Dataset shape:", dataset.shape)
    print("Mean shape:", mean.shape)


    print("Running PCA...")
    [H,W,_] = moises.pca(dataset.T);

    np.save(
        os.path.join(out_dir, 'coeffs.npy'),
        H
    )

    np.save(
        os.path.join(out_dir, 'scores.npy'),
        W
    )

def process_folder(path, out_path, scale, max_frames):

    children = os.listdir(path)
    for child in children:
        c_path = os.path.join(path, child)
        if os.path.isdir(c_path):
            process_folder(
                    c_path,
                    os.path.join(out_path, child),
                    scale, max_frames)
        elif child[-4:] == '.mp4':
            process_video(c_path, max_frames, scale, out_path)

def process_batch(path, out_path, scales=[1], max_frames=[None]):

    for scale in scales:
        for mf in max_frames:
            process_folder(
                    path,
                    os.path.join(out_path, str(scale), str(mf)),
                    scale, mf
            )


process_batch('../videos/regi', 'outt', [.5, .2], [400])
