import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from matplotlib import rc
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.optimize import minimize
import agama

agama.setUnits(mass=1E10, length=1, velocity=1)
mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/analysis/bar_prop/data/'
torque_path = '/n/home01/abeane/starbar/analysis/torques/data/'

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

def read_agama_pot(idx, name, lvl):
    base = '/n/home01/abeane/starbar/analysis/agama_pot/data/'
    fname = base + 'pot_' + name + '-' + lvl + '/pot_' + name + '-' + lvl + '.' + str(idx) + '.txt'
    return agama.Potential(fname)

def read_all_agama_pot(name, lvl):
    base = '/n/home01/abeane/starbar/analysis/agama_pot/data/'
    fbase = base + 'pot_' + name + '-' + lvl + '/pot_' + name + '-' + lvl + '.*.txt'
    nsnap = len(glob.glob(fbase))

    all_agama_pot = []
    for i in range(nsnap):
        all_agama_pot.append(read_agama_pot(i, name, lvl))
    
    return all_agama_pot

def compute_vc(R, pot):
    acc = pot.force(R, 0, 0)
    vcsq = - R * acc[0]
    return np.sqrt(vcsq)

def _to_minimize(R, pot, omega_p):
    vc = compute_vc(R, pot)
    omega = vc / R
    return np.abs(omega - omega_p)

def compute_RCR(pot, omega_p, Rguess=6):
    ans = minimize(_to_minimize, Rguess, args=(pot, omega_p))

    return float(ans.x)

def run():
    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)

    print(bar_prop_Nbody['pattern_speed'][0:300])
    
    agama_pot_Nbody = read_all_agama_pot(Nbody, lvl)
    agama_pot_SMUGGLE = read_all_agama_pot(phS2R35, lvl)

    RCR_Nbody = []
    for i in range(len(bar_prop_Nbody['tlist'])):
        if bar_prop_Nbody['pattern_speed'][i] <= 0.0:
            RCR_Nbody.append(np.nan)
        else:
            RCR = compute_RCR(agama_pot_Nbody[i], bar_prop_Nbody['pattern_speed'][i])
            RCR_Nbody.append(RCR)
    
    RCR_SMUGGLE = []
    for i in range(len(bar_prop_SMUGGLE['tlist'])):
        RCR = compute_RCR(agama_pot_SMUGGLE[i], bar_prop_SMUGGLE['pattern_speed'][i])
        RCR_SMUGGLE.append(RCR)

    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9*cm, 9*cm))

    

    # Second panel, length of bar and mass of bar.
    t300 = bar_prop_Nbody['tlist'][300]

    rot_Nbody = np.array(RCR_Nbody)/bar_prop_Nbody['Rbar']
    rot_SMUGGLE = np.array(RCR_SMUGGLE) / bar_prop_SMUGGLE['Rbar']

    rot_Nbody = savgol_filter(rot_Nbody, 81, 3)
    rot_SMUGGLE = savgol_filter(rot_SMUGGLE, 81, 3)

    ax.plot(bar_prop_Nbody['tlist'] - t300, rot_Nbody, c=tb_c[0], label='N-body')#, ls='dashed')
    ax.plot(bar_prop_SMUGGLE['tlist'], rot_SMUGGLE, c=tb_c[1], label='SMUGGLE')#, ls='dashed')

    print(rot_Nbody[300], RCR_Nbody[300], bar_prop_Nbody['Rbar'][300])

    ax.legend(frameon=False)
    ax.set(ylim=(1.2, 2), ylabel=r'$\mathcal{R}$', xlabel=r'$t\,[\,\textrm{Gyr}\,]$', xlim=(-1.5, 5))

    fig.tight_layout()
    fig.savefig('rot_param.pdf')

    fig, ax = plt.subplots(1, 1, figsize=(9*cm, 9*cm))

    ax.plot(bar_prop_Nbody['tlist'], rot_Nbody, c=tb_c[0], label='N-body')
    ax.legend(frameon=False)
    
    ax.set(ylim=(1, 2), xlim=(0, 8), ylabel=r'$\mathcal{R}$', xlabel=r'$t\,[\,\textrm{Gyr}\,]$')

    fig.tight_layout()
    fig.savefig('rot_param-Nbody.pdf')


if __name__ == '__main__':
    run()

