from tqdm import tqdm

import pickle

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation


def get_vmin_vmax(frames):
    vmin = min(frames, key=lambda x: x.min()).min()
    vmax = max(frames, key=lambda x: x.max()).max()

    return vmin, vmax


def get_animator(frames, vmin, vmax, cmap, size, res):
    fig, ax = plt.subplots(figsize=(size, size), dpi=res//size)

    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    image = ax.imshow(frames[0], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

    def animate(frame):
        image.set_array(frames[frame])

        return image
    
    anim = FuncAnimation(
        fig,
        animate,
        range(len(frames)),
        fargs=[],
        interval=1000 / 25
    )

    return anim

if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Makes the video of frames in the supplied pickle file."
    )

    parser.add_argument(
        "-i",
        "--input",
        help="Name of the input pickle file",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-c",
        "--cmap",
        help="Colour map to use. Default: twilight.",
        required=False,
        type=str,
        default="twilight"
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename",
        required=True,
        type=str,
    )

    args = vars(parser.parse_args())

    with open(args["input"], "rb") as handle:
        frames = pickle.load(handle)

    res = frames[0].shape[0]
    
    vmin, vmax = get_vmin_vmax(frames)

    anim = get_animator(frames, vmin, vmax, cmap=args["cmap"], size=8, res=res)

    anim.save(args["output"], dpi=res // 8)

