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

import agama
agama.setUnits(mass=1E10, length=1, velocity=1)

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
agama_pot_path = '/n/home01/abeane/starbar/analysis/agama_pot/data/'

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def read_snap(idx, name, lvl, parttype=[2, 3, 4], fields=['Coordinates', 'Masses']):
    sn = arepo.Snapshot(snap_path+name+'/'+lvl+'/output', idx, combineFiles=True, parttype=parttype,
                        fields=fields)
    return sn

def read_bar_prop(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = {}
    for key in t['bar_prop'].keys():
        out[key] = t['bar_prop'][key][:]
    
    out['bar_angle'] = t['bar_angle'][:]
    t.close()
    
    return out

def read_agama_pot(idx, name, lvl):
    fname = agama_pot_path + 'pot_' + name + '-' + lvl + '/pot_' + name + '-' + lvl + '.' + str(idx) + '.txt'
    return agama.Potential(fname)

def compute_vcirc(pot):
    Rlist = np.linspace(0, 25, 1000)
    pos = np.array([[R, 0, 0] for R in Rlist])
    acc = pot.force(pos)
    vcsq = - Rlist * acc[:,0]
    return Rlist, np.sqrt(vcsq)

def run():
    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, figsize=(columnwidth*cm, (2./3.)*columnwidth*cm))

    # pot = read_agama_pot(500, Nbody, lvl)
    # R, vc = compute_vcirc(pot)
    # ax.plot(R, vc, c=tb_c[0], label=r'$N$-body')

    
    pot = read_agama_pot(200, phS2R35, lvl)
    R, vc = compute_vcirc(pot)
    ax.plot(R, vc, c=tb_c[0], label=r'$1\,\textrm{Gyr}$')
    
    pot = read_agama_pot(400, phS2R35, lvl)
    R, vc = compute_vcirc(pot)
    ax.plot(R, vc, c=tb_c[1], label=r'$2\,\textrm{Gyr}$')
    
    pot = read_agama_pot(600, phS2R35, lvl)
    R, vc = compute_vcirc(pot)
    ax.plot(R, vc, c=tb_c[2], label=r'$3\,\textrm{Gyr}$')
    
    pot = read_agama_pot(800, phS2R35, lvl)
    R, vc = compute_vcirc(pot)
    ax.plot(R, vc, c=tb_c[3], label=r'$4\,\textrm{Gyr}$')

    # obs_data = np.genfromtxt('eilers_data.txt')
    # ax.errorbar(obs_data[:,0], obs_data[:,1], yerr=obs_data[:,2:].T, c='k', ls = "None")
    # ax.scatter(obs_data[:,0], obs_data[:,1], s=4, c='k', label='Eilers et al. (2019)')

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 400)

    ax.set_title('SMUGGLE')
    ax.set_xlabel(r'$R\,[\,\text{kpc}\,]$')
    ax.set_ylabel(r'$v_{\textrm{circ}}\,[\,\text{km}/\text{s}\,]$')

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('vcirc_time.pdf')
    
def run_init():
    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, figsize=(columnwidth*cm, (2./3.)*columnwidth*cm))

    pot = read_agama_pot(300, Nbody, lvl)
    R, vc = compute_vcirc(pot)
    ax.plot(R, vc, c=tb_c[0], label=r'$t=0\,\textrm{Gyr}$')

    pot = read_agama_pot(0, Nbody, lvl)
    R, vc = compute_vcirc(pot)
    ax.plot(R, vc, c=tb_c[1], label='r$t=-1.5\,\textrm{Gyr}$')

    obs_data = np.genfromtxt('eilers_data.txt')
    ax.errorbar(obs_data[:,0], obs_data[:,1], yerr=obs_data[:,2:].T, c='k', ls = "None")
    ax.scatter(obs_data[:,0], obs_data[:,1], s=4, c='k', label='Eilers et al. (2019)')

    ax.set_xlim(0, 25)
    ax.set_ylim(0, 400)

    ax.set_xlabel(r'$R\,[\,\text{kpc}\,]$')
    ax.set_ylabel(r'$v_{\textrm{circ}}\,[\,\text{km}/\text{s}\,]$')
    
    ax.set_title('$N$-body')

    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig('vcirc_init.pdf')


if __name__ == '__main__':
    run()
    # run_init()

