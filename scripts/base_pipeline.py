import sys, argparse, os
import numpy as np

sys.path.insert(0, "src")

from moises.data import Video, load_video_dataset
from marajomodes.visualization import plot
import moises

def run_base_pipeline(video_path, max_frames=400, scale=1, out_dir="out"):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    video = Video(video_path, max_frames=max_frames, scale=scale)
    video.load()

    t = np.arange(video.frame_count) / video.fps
    freq = np.arange(video.frame_count) * video.fps / video.frame_count

    dataset, mean = load_video_dataset(video)

    print("Dataset shape:", dataset.shape)
    print("Mean shape:", mean.shape)

    X0, X90 = moises.hilbert_augment(dataset)

    print("Running PCA for X0...")
    [H0,W0,V0] = moises.pca(X0.T);
    print("Running PCA for X90...")
    [H90,W90,V90] = moises.pca(X90.T);
    print("PCA completed")

    V = np.concatenate([V0, V90])
    idx = np.argsort(-V)
    V = V[idx]

    H = np.hstack([H0, H90])[:, idx]
    W = np.hstack([W0, W90])[:, idx]

    print("H shape:", H.shape)

    print("Running CP algorithm...")
    num_pc = 16
    unmixed, Wmix = moises.cp_alg(H[:, :num_pc])
    Winvmix = np.flip(np.linalg.inv(Wmix), axis=0)
    unmixed = -np.fliplr(unmixed)

    print("Extracting modal coordinates...")
    srcs = [0, 1, 8, 9, 13, 14]  
    modal_coord, mode_shapes = moises.solve_modal(unmixed, Winvmix, W, srcs, num_pc=num_pc)
    print("modal_coord (espacial):", modal_coord.shape, "— mode_shapes (temporal):", mode_shapes.shape)

    fig = plot.plot_sources(t, freq, unmixed)
    fig.savefig(f"{out_dir}/sources.png")

    fig_mode_shapes = plot.plot_mode_shapes(mode_shapes, srcs, video.width, video.height)
    fig_mode_shapes.savefig(f"{out_dir}/mode_shapes.png")

    fig_modal_coord = plot.plot_modal_coord(modal_coord, t, freq, len(srcs), video.frame_count)
    fig_modal_coord.savefig(f"{out_dir}/modal_coord.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the base pipeline")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--max-frames", type=int, default=400, help="Maximum number of frames to load")
    parser.add_argument("--scale", type=float, default=1, help="Scale factor for the video")
    parser.add_argument("--out-dir", type=str, default="out", help="Output directory")
    args = parser.parse_args()
    run_base_pipeline(args.video_path, args.max_frames, args.scale, args.out_dir)