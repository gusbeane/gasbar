from re import S
import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from matplotlib import rc
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
from numba import njit
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

mpl.use('Agg')

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cmg
cm = 1/2.54

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
fourier_path = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/fourier_sphere/data/'
bprop_path = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/bar_prop/data/'
in_bar_path = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/in_bar/data/'
htrap_path = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/halo_trapped/'

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def run():
    htrap_N = np.load(htrap_path + '/halo_trapped_'+Nbody+'-lvl3.npy', allow_pickle=True).item()
    htrap_S = np.load(htrap_path + '/halo_trapped_'+phS2R35+'-lvl3.npy', allow_pickle=True).item()
    
    time_N = htrap_N['Time']
    ftrap_N = htrap_N['ftrap2']
    
    time_S = htrap_S['Time']
    ftrap_S = htrap_S['ftrap2']

    fig, ax = plt.subplots(1, 1, figsize=(columnwidth*cm, (3./4.)*columnwidth*cm))
    
    ax.plot(time_N-time_N[300], ftrap_N, c=tb_c[0], label=r'$N$-body')
    ax.plot(time_S, ftrap_S, c=tb_c[1], label=r'SMUGGLE')
    
    ax.set(xlabel=r'$t\,[\,\textrm{Gyr}\,]$', ylabel='fraction of halo trapped')
    ax.set(xlim=(0, 5), ylim=(0, 0.001))#, ylim=(0, 0.0008))
    ax.legend(frameon=False)
    
    fig.tight_layout()

    fig.savefig('halo_trapped.pdf')

if __name__ == '__main__':
    in_bar = run()

