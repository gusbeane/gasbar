import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from matplotlib import rc
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter

time_conv = 977.79222168

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/plots/bar_prop/data/'
fourier_path = '/n/home01/abeane/starbar/plots/fourier_component/data/'
torque_path = '/n/home01/abeane/starbar/plots/torques/data/'

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def read_fourier(name, lvl):
    t = h5.File(fourier_path + 'fourier_' + name + '-' + lvl + '.hdf5', mode='r')
    return t

def extract_t_max_A2A0(fourier, debug=False):

    i = 0

    tlist = np.array(fourier['time'])
    A2A0list = []
    while 'snapshot_'+"{:03d}".format(i) in fourier.keys():
        key = 'snapshot_'+"{:03d}".format(i)
        A0 = np.array(fourier[key]['A0'])
        A2r = np.array(fourier[key]['A2r'])
        A2i = np.array(fourier[key]['A2i'])
    
        A2 = np.sqrt(A2r*A2r + A2i*A2i)
        A2A0list.append(np.max(A2/A0))
        if debug:
            print(tlist[i], fourier[key]['Rlist'][np.argmax(A2/A0)], np.max(A2/A0))
        i += 1
    
    return np.array(tlist)/time_conv, np.array(A2A0list)

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def run():
    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, figsize=(9*cm, 7*cm))
    
    fourier = read_fourier(Nbody, lvl)
    t, A2A0 = extract_t_max_A2A0(fourier)
    ax.plot(t-t[300], savgol_filter(A2A0, 81, 3), c=tb_c[0], label=r'$N$-body')

    fourier = read_fourier(phS2R35, lvl)
    t, A2A0 = extract_t_max_A2A0(fourier)
    ax.plot(t, savgol_filter(A2A0, 81, 3), c=tb_c[1], label='SMUGGLE')

    ax.set(xlim=(0, 5), ylim=(0, 0.7))
    ax.set(xlabel=r'$t\,[\,\textrm{Gyr}\,]$', ylabel=r'$\textrm{max}\left(\left|A_2/A_0\right|\right)$')
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('fig-A2.pdf')


if __name__ == '__main__':
    run()
