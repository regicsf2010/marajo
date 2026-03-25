import os

def process_folder(path, out_path, scale, max_frames, npc):

    children = os.listdir(path)
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
        elif child == 'coeffs.npy':
            print(c_path)

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

process_batch(
    'out/raw/',
    'out/plot/',
)
