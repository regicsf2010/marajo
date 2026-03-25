import argparse, os, sys

sys.path.insert(0, "src")

import moises
import numpy as np


def run_cp(
    path: str,
    npc: int = 16,
    out_dir: str | None = None,
    *,
    use_cuda: bool | None = None,
):

    print("Running cp algorithm over", path)

    coeffs = np.load(path)

    unmixed, Wmix = moises.cp_alg(coeffs[:, :npc], use_cuda=use_cuda)
    Winvmix = np.flip(np.linalg.inv(Wmix), axis=0)
    unmixed = -np.fliplr(unmixed)

    if out_dir is None:
        out_dir = '/'.join(path.split('/')[:-1])


    np.save(
        os.path.join(out_dir, 'unmixed.npy'),
        unmixed
    )
    np.save(
        os.path.join(out_dir, 'winvmix.npy'),
        Winvmix
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CP decomposition on coefficient array (.npy).",
        epilog="Example batch parallelism: "
        "find out/raw -name 'coeffs.npy' -print0 | xargs -0 -P8 -n1 python scripts/cp_runner.py --cuda",
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to the .npy file (rows x features; first npc columns are used).",
    )
    parser.add_argument(
        "--npc",
        type=int,
        default=16,
        help="Number of principal components / columns to take from coeffs (default: 16).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory for unmixed.npy and winvmix.npy. Default: same directory as the input file.",
    )
    cuda_group = parser.add_mutually_exclusive_group()
    cuda_group.add_argument(
        "--cuda",
        action="store_true",
        help="Force CuPy/CUDA (fails if unavailable).",
    )
    cuda_group.add_argument(
        "--cpu",
        action="store_true",
        help="Force NumPy/CPU only.",
    )
    args = parser.parse_args()
    use_cuda: bool | None
    if args.cuda:
        use_cuda = True
    elif args.cpu:
        use_cuda = False
    else:
        use_cuda = None
    run_cp(args.path, npc=args.npc, out_dir=args.out_dir, use_cuda=use_cuda)


