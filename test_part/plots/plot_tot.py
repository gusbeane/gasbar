import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from matplotlib import rc
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
import re
import astropy.units as u
import matplotlib.pylab as pl

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/analysis/bar_prop/data/'
torque_path = '/n/home01/abeane/starbar/analysis/torques/data/'

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']



# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'
phS15R35 = 'phantom-vacuum-Sg15-Rc3.5'
phS10R35 = 'phantom-vacuum-Sg10-Rc3.5'
phS05R35 = 'phantom-vacuum-Sg05-Rc3.5'

lvl = 'lvl3'
lvlGFM = 'lvl3-GFM'

def read_Gout(I):
    fname = '../runs/output-Gvar/out_G'+str(I)+'.npy'
    t = np.load(fname, allow_pickle=True).tolist()
    return t

def read_Iout(I):
    fname = '../runs/output-Ivar/out_I'+str(I)+'.npy'
    t = np.load(fname, allow_pickle=True).tolist()
    return t

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

def run():
    # print_gas_fractions()
    phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'
    Nbody = 'Nbody'
    lvl='lvl3'
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)
    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    
    cm = 1/2.54
    
    fig, ax = plt.subplots(1, 1, figsize=(columnwidth*cm, (3./4.)*columnwidth*cm))
    
    G = 20
    I = 8

    # colors = pl.cm.cool(np.linspace(0,1,n))
    
    #cmap = mpl.colors.LinearSegmentedColormap.from_list("", [tb_c[0], tb_c[1]])
    #colors = cmap(np.linspace(0, 1, N))
    
    outN = read_Iout(I)
    outS = read_Gout(G)

    timeN = outN['time']
    timeS = outS['time']
    
    psN = outN['ps']
    psS = outS['ps']

    ax.plot(timeN, psN, c=tb_c[0], ls='dashed')
    ax.plot(timeS, psS, c=tb_c[1], ls='dashed')
        
    t = bar_prop_Nbody['tlist']
    ps_N = savgol_filter(bar_prop_Nbody['pattern_speed'], 81, 3)
    ax.plot(t - t[300], ps_N, c=tb_c[0], label=r'$N$-body')
    
    t = bar_prop_SMUGGLE['tlist']
    ps_S = savgol_filter(bar_prop_SMUGGLE['pattern_speed'], 81, 3)
    ax.plot(t - t[300], ps_S, c=tb_c[1], label='SMUGGLE')
    
    ax.set(ylim=(0, 60), ylabel=r'$\Omega_p\,[\,\text{km}/\text{s}/\text{kpc}\,]$')
    ax.set(xlim=(0, 5), xlabel=r'$t\,[\,\textrm{Gyr}\,]$')
    
    ax.text(2, 23, r'$\tau_{\textrm{gas}} = 0$', c=tb_c[0])
    ax.text(3, 45, r'$\tau_{\textrm{gas}} = 20$', c=tb_c[1])

    ax.legend(frameon=False, loc='lower left')

    fig.tight_layout()

    fig.savefig('sam_tot.pdf')

if __name__ == '__main__':
    run()

