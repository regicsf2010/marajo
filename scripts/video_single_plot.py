import argparse
import os
import sys

import numpy as np

sys.path.insert(0, "src")

from marajomodes.visualization import plot



def plot_single_sources(path, out_dir = 'out/processed'):

    # Here we assume a path like prefix/{scale}/{n_frames}/{date}/{fps}/rotation_{c}/{name}.npy

    print('Running over', path)

    unmixed = np.load(path)
    pieces = path.split('/')
    out_dir = os.path.join(out_dir, *pieces[-6:-1])
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fps = pieces[-3]
    if 'fps' in fps: fps = fps[:-3] 
    fps = int(fps)
    frame_count = int(pieces[-5])

    t = np.arange(frame_count) / fps
    freq = np.arange(frame_count) * fps / frame_count

    # Precompute FFT features for all sources at once (optionally on CUDA),
    # then pass them into the plotting function to avoid per-source FFT loops.
    print('Calculating fft')
    psd_pos, phase_pos = plot.compute_fft_features_cuda(unmixed, use_cuda=None)

    print('plotting large')
    fig = plot.plot_sources_large(t, freq, unmixed, spectra=(psd_pos, phase_pos))
    fig.savefig(f"{out_dir}/sources_large.png")

    print('plotting simple')
    fig_simple = plot.plot_source_simple(t, freq, unmixed, spectra=(psd_pos, phase_pos))
    fig_simple.savefig(f"{out_dir}/sources_simple.png")

    print('writing csv')
    plot.write_sources_top_peaks_csv(t, freq, spectra=(psd_pos, phase_pos), 
        csv_path=f"{out_dir}/peaks.csv", top_k=20
    )

    print('Finished', path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot sources (large + simple) and write peaks CSV from unmixed.npy."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to unmixed.npy (expected under .../{fps}/rotation_{k}/unmixed.npy).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="out/processed",
        help="Base output directory (default: out/processed).",
    )
    args = parser.parse_args()
    plot_single_sources(args.path, out_dir=args.out_dir)
