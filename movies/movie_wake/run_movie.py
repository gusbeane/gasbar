from make_movie import make_movie
import sys
import os

base = sys.argv[1]
lvl  = sys.argv[2]

width = 30.0
nres = 256

vmin = -0.004
vmax = 0.004

data_dir = 'data/'

name = base + '-' + lvl + '_w' + "{:.01f}".format(width) + '_n' + str(nres)
proj = data_dir + name + '.hdf5'

mov_wake = 'movies/' + name + '_wake_xy.mp4'

make_movie(proj, 1, 'xy', mov_wake, vmin=vmin, vmax=vmax, plot_time=True)
