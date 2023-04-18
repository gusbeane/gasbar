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

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

basepath = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/'

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
fourier_path = basepath + '/analysis/fourier_component/data/'
bprop_path = basepath + '/analysis/bar_prop/data/'

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

    return out

def run():
    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)
    print(bar_prop_SMUGGLE)

    t300 = bar_prop_Nbody['tlist'][300]

    cm = 1/2.54
    fig, ax = plt.subplots(2, 1, figsize=(columnwidth*cm, columnwidth*cm), sharex=True)
    fig_us, ax_us = plt.subplots(2, 1, figsize=(columnwidth*cm, columnwidth*cm), sharex=True)
    
    # First panel, length of bar and mass of bar.
    #ax[0].plot(bar_prop_Nbody['tlist'] - t300, savgol_filter(bar_prop_Nbody['Rbar'], 81, 3), c=tb_c[0],
    #            label=r'$N$-body')
    #ax[0].plot(bar_prop_SMUGGLE['tlist'], savgol_filter(bar_prop_SMUGGLE['Rbar'], 81, 3), c=tb_c[1],
    #            label='SMUGGLE')
    
    ax_us[0].plot(bar_prop_Nbody['tlist'] - t300, bar_prop_Nbody['Rbar'], c=tb_c[0],
                label=r'$N$-body')
    ax_us[0].plot(bar_prop_SMUGGLE['tlist'], bar_prop_SMUGGLE['Rbar'], c=tb_c[1],
                label='SMUGGLE')

    ax[0].set(ylim=(0, 7), ylabel=r'$R_{\text{bar}}\,[\,\text{kpc}\,]$')
    ax[0].legend(frameon=False)
    
    ax_us[0].set(ylim=(0, 7), ylabel=r'$R_{\text{bar}}\,[\,\text{kpc}\,]$')
    ax_us[0].legend(frameon=False)

    fourier = read_fourier(Nbody, lvl)
    t, A2A0 = extract_t_max_A2A0(fourier)
    ax[1].plot(t-t[300], savgol_filter(A2A0, 81, 3), c=tb_c[0], label=r'$N$-body')
    ax_us[1].plot(t-t[300], A2A0, c=tb_c[0], label=r'$N$-body')

    fourier = read_fourier(phS2R35, lvl)
    t, A2A0 = extract_t_max_A2A0(fourier)
    ax[1].plot(t, savgol_filter(A2A0, 81, 3), c=tb_c[1], label='SMUGGLE')
    ax_us[1].plot(t, A2A0, c=tb_c[1], label='SMUGGLE')

    
    ax[1].set(xlim=(0, 5), ylim=(0, 0.7))
    ax[1].set(xlabel=r'$t\,[\,\textrm{Gyr}\,]$', ylabel=r'$\textrm{max}\left(\left|A_2/A_0\right|\right)$')
    
    ax_us[1].set(xlim=(0, 5), ylim=(0, 0.7))
    ax_us[1].set(xlabel=r'$t\,[\,\textrm{Gyr}\,]$', ylabel=r'$\textrm{max}\left(\left|A_2/A_0\right|\right)$')

    fig.tight_layout()
    fig.savefig('Rb_A2.pdf')
    
    fig_us.tight_layout()
    fig_us.savefig('Rb_A2_unsmoothed.pdf')


if __name__ == '__main__':
    run()
