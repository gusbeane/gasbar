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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/plots/bar_prop/data/'
torque_path = '/n/home01/abeane/starbar/plots/torques/data/'

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def fix_bar_angle(bar_angle):
    out = np.zeros(len(bar_angle))
    out[0] = bar_angle[0]

    for i in range(1, len(bar_angle)):
        dphi = bar_angle[i] - bar_angle[i-1]
        if dphi < -np.pi:
            dphi += 2.*np.pi
        
        out[i] = out[i-1] + dphi
    
    return out

def read_bar_prop(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = {}
    for key in t['bar_prop'].keys():
        out[key] = t['bar_prop'][key][:]
    
    out['bar_angle'] = fix_bar_angle(t['bar_angle'][:])
    t.close()


    # print(out)
    # print(len(out['bar_angle']))
    # print(len(out['tlist']))

    out['pattern_speed'] = np.gradient(out['bar_angle'], out['tlist'])


    return out

def read_torque(name, lvl):
    base = torque_path + 'torques_' + name + '-' + lvl + '/torques_' + name + '-' + lvl + '.'

    # nfiles = len(glob.glob(base + '*.hdf5'))
    nfiles = 1200

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
    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(3.5, 6))

    # First panel, pattern speed.
    t300 = bar_prop_Nbody['tlist'][300]
    ax[0].plot(bar_prop_Nbody['tlist'] - t300, bar_prop_Nbody['pattern_speed'], c=tb_c[0], label='N-body')
    ax[0].plot(bar_prop_SMUGGLE['tlist'], bar_prop_SMUGGLE['pattern_speed'], c=tb_c[1], label='SMUGGLE')

    ax[0].set(ylim=(0, 60), ylabel=r'$\Omega_p\,[\,\text{km}/\text{s}/\text{kpc}\,]$')
    ax[0].set(xlim=(0, 5))

    ax[0].legend(frameon=False)

    # Second panel, length of bar and mass of bar.
    ax[1].plot(bar_prop_Nbody['tlist'] - t300, bar_prop_Nbody['Rbar'], c=tb_c[0])
    ax[1].plot(bar_prop_SMUGGLE['tlist'], bar_prop_SMUGGLE['Rbar'], c=tb_c[1])

    ax[1].set(ylim=(0, 7), ylabel=r'$R_{\text{bar}}\,[\,\text{kpc}\,]$')

    # Third panel, torques.
    tlist, tz_halo, tz_not_bar, _= read_torque(Nbody, lvl)
    tlist_g, tz_halo_g, tz_not_bar_g, tz_gas_g = read_torque(phS2R35, lvl)

    tz_halo = savgol_filter(tz_halo, 81, 3)
    tz_halo_g = savgol_filter(tz_halo_g, 81, 3)
    tz_gas_g = savgol_filter(tz_gas_g, 81, 3)

    ax[2].plot(tlist-tlist[300], -tz_halo, c=tb_c[0])
    ax[2].plot(tlist_g, -tz_halo_g, c=tb_c[1])
    ax[2].plot(tlist_g, -tz_gas_g, c=tb_c[1], ls='dashed')

    ax[2].axhline(0, c='k')

    ax[2].set(ylim=(-100, 100), ylabel=r'$\tau_{\text{on bar}}\,[\,10^{10}\,M_{\odot}\,(\text{km}/\text{s})^2\,]$')
    ax[2].set_xlabel(r'$t\,[\,\text{Gyr}\,]$')

    custom_lines = [mpl.lines.Line2D([0], [0], color='k'),
                    mpl.lines.Line2D([0], [0], color='k', ls='dashed')]
    ax[2].legend(custom_lines, ['halo', 'gas'], frameon=False)


    fig.tight_layout()

    fig.savefig('fig2.pdf')

if __name__ == '__main__':
    run()

