import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from matplotlib import rc
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/analysis/bar_prop/data/'
torque_path = '/n/home01/abeane/starbar/analysis/torques-rot/data/'

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def read_torque(name, lvl):
    base = torque_path + 'torques_' + name + '-' + lvl + '/torques_' + name + '-' + lvl + '.'

    nfiles = len(glob.glob(base + '*.hdf5'))
    # nfiles = 1200

    tz_halo = []
    tz_not_bar = []
    tz_gas = []

    tlist = []

    for i in tqdm(range(nfiles)):
        fname = base + str(i) + '.hdf5'
        t = h5.File(fname, mode='r')

        torque_halo = t['total_torques'].attrs['halo']
        tz_halo.append(torque_halo[2])

        torque_not_bar = t['total_torques'].attrs['not_bar']
        tz_not_bar.append(torque_not_bar[2])
    
        if 'gas' in t['total_torques'].attrs.keys():
            torque_gas = t['total_torques'].attrs['gas']
            tz_gas.append(torque_gas[2])
        
        tlist.append(t['parameters'].attrs['Time'])
        
        t.close()
    
    tz_halo = np.array(tz_halo)
    tz_not_bar = np.array(tz_not_bar)
    tz_gas = np.array(tz_gas)
    tlist = np.array(tlist)

    return tlist, tz_halo, tz_not_bar, tz_gas

def run():
    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(columnwidth*cm, (2./3.)*columnwidth*cm))

    # First panel, pattern speed.
    tlist_30, tz_halo_30, _, tz_gas_30 = read_torque(phS2R35, 'lvl3-rot30')
    tlist_35, tz_halo_35, _, tz_gas_35 = read_torque(phS2R35, 'lvl3-rot35')
    tlist_40, tz_halo_40, _, tz_gas_40 = read_torque(phS2R35, 'lvl3-rot40')

    rot_list = np.array([30, 35, 40])

    start = 50
    end = 250

    mean_tz_list = []
    mean_tz_list.append(np.mean(tz_gas_30[start:end]))
    mean_tz_list.append(np.mean(tz_gas_35[start:end]))
    mean_tz_list.append(np.mean(tz_gas_40[start:end]))
    mean_tz_list = np.array(mean_tz_list)

    ax.plot(rot_list, -mean_tz_list, c='k')
    ax.scatter(rot_list, -mean_tz_list, c='k', s=25)

    # ax.axhline(0, c='k', ls='dashed')

    ax.set(xlim=(29, 41), ylim=(0, 25))
    ax.set(xticks=rot_list)
    ax.set(xlabel=r'$\Omega_p\,[\,\textrm{km}/\textrm{s}/\textrm{kpc}\,]$')
    ax.set(ylabel=r'$\left< \tau_{\textrm{on bar}} \right>\,[\,10^{10}\,M_{\odot}\,(\text{km}/\text{s})^2\,]$')

    fig.tight_layout()

    fig.savefig('torque_ps.pdf')

if __name__ == '__main__':
    run()

