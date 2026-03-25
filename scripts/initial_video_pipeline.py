import sys, os
import numpy as np

sys.path.insert(0, "src")

from moises.data import Video, load_video_dataset
import moises

def process_video(video_path, max_frames, scale=1, out_dir="prep", npc=16):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print(f"Processing video from {video_path}")
    print("Scale", scale)
    print("Max Frames", max_frames)
    print("npc (PCA cap for .npy)", npc)

    video = Video(video_path, max_frames=max_frames, scale=scale, workers=6)
    video.load()

    dataset, mean = load_video_dataset(video)

    print("Dataset shape:", dataset.shape)
    print("Mean shape:", mean.shape)


    print("Running PCA...")
    [H, W, _] = moises.pca(dataset.T)

    k = min(npc, H.shape[1])
    H_save = H[:, :k]
    W_save = W[:, :k]

    np.save(
        os.path.join(out_dir, 'coeffs.npy'),
        H_save,
    )

    np.save(
        os.path.join(out_dir, 'scores.npy'),
        W_save,
    )

def process_folder(path, out_path, scale, max_frames, npc):

    children = os.listdir(path)
    rotation = 1
    for child in children:
        c_path = os.path.join(path, child)
        if os.path.isdir(c_path):
            process_folder(
                c_path,
                os.path.join(out_path, child),
                scale,
                max_frames,
                npc,
            )
        elif child[-4:] == '.mp4':
            process_video(
                    c_path, 
                    max_frames, 
                    scale, 
                    out_dir= os.path.join(out_path, f'rotation_{rotation}'), 
                    npc=npc
            )
            rotation += 1

def process_batch(path, out_path, scales=[1], max_frames=[None], npc=16):

    for scale in scales:
        for mf in max_frames:
            process_folder(
                path,
                os.path.join(out_path, str(scale), str(mf)),
                scale,
                mf,
                npc,
            )


process_batch('../videos/regi', 'out/raw', [.5, .3, .2], [500, 400], npc=16)
