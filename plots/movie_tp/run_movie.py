from make_movie import make_movie
import sys
import os

base = sys.argv[1]
lvl  = sys.argv[2]

width = 30.0
nres = 256

star_vmin = 1E-3
star_vmax = 1E0
# gas_vmin  = 0.1 * star_vmin
# gas_vmax  = 0.1 * star_vmax

data_dir = 'data/'

name = base + '-' + lvl + '_w' + "{:.01f}".format(width) + '_n' + str(nres)
proj = data_dir + name + '.hdf5'

mov_starxy = 'movies/' + name + '_star_xy.mp4'
mov_starxz = 'movies/' + name + '_star_xz.mp4'
mov_testxy = 'movies/' + name + '_test_xy.mp4'
mov_testxz = 'movies/' + name + '_test_xz.mp4'
mov_4panel = 'movies/' + name + '_4panel.mp4'

make_movie(proj, [2, 3], 'xy', mov_starxy, vmin=star_vmin, vmax=star_vmax, plot_time=True)
make_movie(proj, [2, 3], 'xz', mov_starxz, vmin=star_vmin, vmax=star_vmax)
make_movie(proj, 5, 'xy', mov_testxy, vmin=star_vmin, vmax=star_vmax)
make_movie(proj, 5, 'xz', mov_testxz, vmin=star_vmin, vmax=star_vmax, plot_time=True)

if os.path.exists('movies/' + name + '_4panel.mp4'):
    os.remove('movies/' + name + '_4panel.mp4')

ffmpeg_call = 'ffmpeg -i ' + mov_starxz + ' -i ' + mov_testxz + ' -i ' + mov_starxy + ' -i ' + mov_testxy
ffmpeg_call = ffmpeg_call + ' -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[out]" -map "[out]" '
ffmpeg_call = ffmpeg_call + mov_4panel

os.system(ffmpeg_call)

