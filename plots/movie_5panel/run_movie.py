from make_movie import make_movie
import sys
import os

base = sys.argv[1]
lvl  = sys.argv[2]

width = 30.0
nres = 256

data_dir = 'data/'

name = base + '-' + lvl + '_w' + "{:.01f}".format(width) + '_n' + str(nres)
proj = data_dir + name + '.hdf5'

mov_gasxy = 'movies/' + name + '_gas_xy.mp4'
mov_gasxz = 'movies/' + name + '_gas_xz.mp4'
mov_starxy = 'movies/' + name + '_star_xy.mp4'
mov_starxz = 'movies/' + name + '_star_xz.mp4'
mov_4panel = 'movies/' + name + '_4panel.mp4'

make_movie(proj, 0, 'xy', mov_gasxy)
make_movie(proj, 0, 'xz', mov_gasxz)
make_movie(proj, [2, 3, 4], 'xy', mov_starxy)
make_movie(proj, [2, 3, 4], 'xz', mov_starxz)

if os.path.exists('movies/' + name + '_4panel.mp4'):
    os.remove('movies/' + name + '_4panel.mp4')

ffmpeg_call = 'ffmpeg -i ' + mov_gasxz + ' -i ' + mov_starxz + ' -i ' + mov_gasxy + ' -i ' + mov_starxy
ffmpeg_call = ffmpeg_call + ' -filter_complex "[0:v][1:v]hstack[top];[2:v][3:v]hstack[bottom];[top][bottom]vstack[out]" -map "[out]" '
ffmpeg_call = ffmpeg_call + mov_4panel

os.system(ffmpeg_call)

