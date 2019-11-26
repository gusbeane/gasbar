#from swiftsimio import load
import sys
mach = sys.platform
if mach == 'darwin':
    sys.path.append('/Users/abeane/scratch/mwib_analysis')

# from mwib_analysis import mwib_io
import h5py as h5
from sphviewer.tools import QuickView

from unyt import msun, kpc
from tqdm import tqdm

import pickle

import numpy as np


def grab_frame(filename, particle_type, res, fixed_hsml, center=None, extent=None, max_smooth=None):
    # snap = mwib_io.read_snap(filename)
    f = h5.File(filename, mode='r')
    particle_group = {}
    for k in ['Coordinates', 'Masses', 'Density']:
        particle_group[k] = np.array(f[particle_type][k])

    fh = f['Parameters'].attrs['GasSoftFactor']
    volume = np.divide(particle_group['Masses'], particle_group['Density'])
    particle_group['Smoothing_length'] = np.multiply(fh, np.power(np.multiply(volume, 3/(4.*np.pi)), 1./3.))

    # convert masses to Msun
    particle_group['Masses'] = np.multiply(particle_group['Masses'], 1E10)

    if center is None:
        center = [0, 0, 0]

    # particle_group = snap[particle_type]
    # particle_group = getattr(snap, particle_type)

    try:
        coordinates = particle_group['Coordinates']
    except AttributeError:
        # Dataset doesn't exist
        print('Dataset doesnt exist')
        return np.zeros((res, res), dtype=float)

    masses = particle_group['Masses']

    try:
        # hsml = particle_group.smoothing_length.to(kpc).value
        hsml = particle_group['Smoothing_length']
        max_hsml = max_smooth
    except:
        print("Cannot find smoothing length; generating")
        if fixed_hsml:
            max_hsml = fixed_hsml
            hsml = None
        else:
            hsml = None
            max_hsml = None

    out = QuickView(
        coordinates,
        mass=masses,
        hsml=hsml,
        r="infinity",
        xsize=res,
        ysize=res,
        plot=False,
        max_hsml=max_hsml,
        x=center[0], y=center[1], z=center[2],
        extent=extent
    ).get_image()


    return out


def grab_all_frames(stub, start, stop, particle_type, res, fixed_hsml, center, extent, max_smooth):
    return [
        grab_frame("{}_{:03d}.hdf5".format(stub, x), particle_type, res, fixed_hsml, center, extent, max_smooth=max_smooth)
        for x in tqdm(range(start, stop + 1))
    ]


if __name__ == "__main__":
    import argparse as ap

    parser = ap.ArgumentParser(
        description="Makes the frames of particle type from a simulation with starting path stub and writes the output to output_filename at resolution resxres."
    )

    parser.add_argument(
        "-s",
        "--stub",
        help="Start of the filename (e.g. ./santabarbara for snapshots in the current working directory called santabarbara_xxxx.hdf5",
        type=str,
        required=True,
    )

    parser.add_argument(
        "-p",
        "--ptype",
        help="Particle type to make a movie of. Can be either gas, stars, or dark_matter.",
        required=True,
        type=str,
    )

    parser.add_argument(
        "-i",
        "--initial",
        help="Initial snapshot to start on. Default: 0",
        default=0,
        type=int,
        required=False,
    )

    parser.add_argument(
        "-f",
        "--final",
        help="Final snapshot number to end on.",
        type=int,
        required=True,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="Output filename. Default: '--ptype'.pickle.",
        required=False,
        default="DEFAULT",
        type=str,
    )

    parser.add_argument(
        "-r",
        "--res",
        help="Resolution to make the movie at. Default = 2048",
        required=False,
        type=int,
        default=2048,
    )

    parser.add_argument(
        "-k",
        "--fixed_hsml",
        help="Set a fixed smoothing length (in kpc) for all particles.",
        required=False,
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "-m",
        "--max_smooth",
        help="Maximum smoothing length used.",
        required=False,
        type=float,
        default=1.0,
    )
    
    parser.add_argument(
        "-x",
        help="x position for center of simulation.",
        required=False,
        type=float,
        default=0.0,
    )
    
    parser.add_argument(
        "-y",
        help="y position for center of simulation.",
        required=False,
        type=float,
        default=0.0,
    )
    
    parser.add_argument(
        "-z",
        help="z position for center of simulation.",
        required=False,
        type=float,
        default=0.0,
    )
    
    parser.add_argument(
        "-w",
        "--width",
        help="Width of frame.",
        required=False,
        type=float,
        default=10.0,
    )

    args = vars(parser.parse_args())

    # Parameters
    particle_type = args["ptype"]
    stub = args["stub"]
    start = args["initial"]
    stop = args["final"]
    res = args["res"]
    fixed_hsml = args["fixed_hsml"]

    x = args["x"]
    y = args["y"]
    z = args["z"]
    w = args["width"]

    max_smooth = args["max_smooth"]

    center = [x, y, z]
    extent = [-w, w, -w, w]

    if args["output"] is not "DEFAULT":
        output_filename = args["output"]
    else:
        output_filename = "{}.pickle".format(particle_type)

    frames = grab_all_frames(stub, start, stop, particle_type, res, fixed_hsml, center, extent, max_smooth)

    print("Dumping to {}".format(output_filename))

    with open(output_filename, "wb") as handle:
        pickle.dump(frames, handle)

    exit(0)
