import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py as h5
from tqdm import tqdm

from matplotlib.animation import FuncAnimation

class animate_maker(object):
    def __init__(self, proj_hdf5, projection, parttype, nres):
        self.proj_hdf5 = proj_hdf5
        self.projection = projection
        self.parttype = parttype
        self.nres = nres
        self.base_key_list = ['PartType' + str(pt) + '/' + projection + '/snapshot_' for pt in self.parttype]

    def __call__(self, frame, im):
        heatmap = np.zeros((self.nres, self.nres))
        for base_key in self.base_key_list:
            this_ht = self.proj_hdf5[base_key + "{:03d}".format(frame)][:]
            heatmap += this_ht

        im.set_data(heatmap.T)
        return (im,)


def make_movie(projection_file, parttype, projection, fout, vmin=1E-3, vmax=1E-1, fps=16):
    assert projection in ['xy', 'xz', 'yz']

    if isinstance(parttype, int):
        parttype = [parttype]

    f = h5.File(projection_file, mode='r')

    width = f.attrs['width']
    nres = f.attrs['nres']
    maxsnap = f.attrs['maxsnap']

    # initialize fig and ax, remove boundary
    fig, ax = plt.subplots(1, figsize=(2, 2))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")

    # initialize im
    extent = [-width/2.0, width/2.0, -width/2.0, width/2.0]
    im = ax.imshow(np.full((nres, nres), vmin), extent=extent, origin='lower', 
                   norm=mpl.colors.LogNorm(), vmin=vmin, vmax=vmax)

    # initialize animator
    animate = animate_maker(f, projection, parttype, nres)

    animation = FuncAnimation(fig, animate, tqdm(np.arange(maxsnap+1)), fargs=[im], interval=1000 / fps)
    animation.save(fout, dpi=nres)

if __name__ == '__main__':
    make_movie('data/fid-dispPoly-fg0.1-lvl5_w30.0_n256.hdf5', 0, 'xy', 'test_gas.mp4')
    make_movie('data/fid-dispPoly-fg0.1-lvl5_w30.0_n256.hdf5', [2, 3, 4], 'xy', 'test_star.mp4')
