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

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
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
phS15R35 = 'phantom-vacuum-Sg15-Rc3.5'
phS10R35 = 'phantom-vacuum-Sg10-Rc3.5'
phS08R35 = 'phantom-vacuum-Sg08-Rc3.5'
phS05R35 = 'phantom-vacuum-Sg05-Rc3.5'

lvl = 'lvl3'

def read_fourier(name, lvl, basepath='/n/home01/abeane/starbar/plots/'):
    f = h5.File(basepath+'/fourier_component/data/fourier_'+name+'-'+lvl+'.hdf5', mode='r')
    return f

def read_snap(idx, name, lvl, parttype=[0, 2, 3, 4], fields=['Coordinates', 'Masses']):
    sn = arepo.Snapshot(snap_path+name+'/'+lvl+'/output', idx, combineFiles=True, parttype=parttype,
                        fields=fields)
    return sn

def get_sorted_keys(dat):
    keys = list(dat.keys())
    # only keep keys that are snapshot keys
    keys = [k for k in keys if 'snapshot' in k]

    # extract and sort indices
    indices = [int(re.findall(r'\d?\d?\d\d\d', k)[0]) for k in keys]
    sorted_arg = np.argsort(indices)
    keys_sorted = [keys[i] for i in sorted_arg]

    return keys_sorted

def get_A2_angle(dat, keys, Rbin):
    Rlist = np.array([np.array(dat[k]['Rlist']) for k in keys])
    A2r = np.array([np.array(dat[k]['A2r']) for k in keys])
    A2i = np.array([np.array(dat[k]['A2i']) for k in keys])
    phi = np.arctan2(A2i, A2r)
    phi = phi[:,Rbin]
    R_at_Rbin = Rlist[:,Rbin]
    
    time = np.array(dat['time'])

    return time, R_at_Rbin, phi

def get_bar_angle(phi, firstkey):
    out = np.zeros(len(phi))

    # set the first bar angle
    first_bar_angle = phi[firstkey]/2.0
    out[firstkey] = first_bar_angle
    
    # set all subsequent angles
    for i in np.arange(firstkey+1, len(out)):
        dphi = phi[i] - phi[i-1]
        if dphi < -np.pi:
            dphi += 2.*np.pi
        out[i] = out[i-1] + dphi/2.0

    # set all previous angles to be the bar angle
    for i in np.arange(0, firstkey):
        out[i] = first_bar_angle

    return out

def main_bar_angle(dat, Rbin = 5, firstkey = 150, nmax = 10):
    keys = get_sorted_keys(dat)
    time, R, phi = get_A2_angle(dat, keys, Rbin)
    bar_angle = get_bar_angle(phi, firstkey)

    pattern_speed = np.gradient(bar_angle, time) / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    time = time * u.Myr 
    time = time.to_value(u.kpc / (u.km/u.s))

    return time, pattern_speed

def get_pattern_speed(name, lvl):
    fourier = read_fourier(name, lvl)

    if 'Nbody' in name:
        firstkey = 150
    else:
        firstkey = 0

    time, pattern_speed = main_bar_angle(fourier, firstkey=firstkey)
    return time, pattern_speed

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def print_gas_fractions():
    for name in [phS05R35, phS10R35, phS15R35, phS2R35]:
        sn = read_snap(0, name, lvl)
        
        ptype_mass = sn.NumPart_Total * sn.MassTable
        Mg = np.sum(sn.part0.mass.value)
        Ms = ptype_mass[2] + ptype_mass[3]
        fg = Mg / (Mg + Ms)
        print('for name', name, 'gas fraction is', fg)

def run():
    # print_gas_fractions()

    time_Nbody, ps_Nbody = get_pattern_speed(Nbody, lvl)
    time_SMUGGLE20, ps_SMUGGLE20 = get_pattern_speed(phS2R35, lvl)
    time_SMUGGLE15, ps_SMUGGLE15 = get_pattern_speed(phS15R35, lvl)
    time_SMUGGLE10, ps_SMUGGLE10 = get_pattern_speed(phS10R35, lvl)
    time_SMUGGLE08, ps_SMUGGLE08 = get_pattern_speed(phS08R35, lvl)
    time_SMUGGLE05, ps_SMUGGLE05 = get_pattern_speed(phS05R35, lvl)

    # for ps in [ps_Nbody, ps_SMUGGLE20, ps_SMUGGLE15, ps_SMUGGLE10, ps_SMUGGLE05]:
        # ps = savgol_filter(ps, 81, 3)

    cm = 1/2.54

    fig, ax = plt.subplots(1, 1, figsize=(9*cm, 9*cm))

    # First panel, pattern speed.

    lw = 1
    ax.plot(time_SMUGGLE20, savgol_filter(ps_SMUGGLE20, 81, 3), c=tb_c[1], label=r'$20$ (fiducial)')
    ax.plot(time_SMUGGLE15, savgol_filter(ps_SMUGGLE15, 81, 3), c=tb_c[2], label=r'$15$')
    ax.plot(time_SMUGGLE10, savgol_filter(ps_SMUGGLE10, 81, 3), c=tb_c[3], label=r'$10$')
    # ax.plot(time_SMUGGLE08, savgol_filter(ps_SMUGGLE08, 81, 3), c=tb_c[4], label=r'$8$')
    ax.plot(time_SMUGGLE05, savgol_filter(ps_SMUGGLE05, 81, 3), c=tb_c[5], label=r'$5$')

    ax.plot(time_Nbody - time_Nbody[300], savgol_filter(ps_Nbody, 81, 3), lw=lw, c=tb_c[0], label=r'0 ($N$-body)')

    ax.set(ylim=(0, 60), ylabel=r'$\Omega_p\,[\,\text{km}/\text{s}/\text{kpc}\,]$')
    ax.set(xlim=(0, 5), xlabel=r'$t\,[\,\textrm{Gyr}\,]$')

    ax.legend(frameon=False, title=r'$\Sigma_{\textrm{gas}}\,[\,M_{\odot}/\textrm{kpc}^2\,]$')

    fig.tight_layout()

    fig.savefig('fig-fgas.pdf')

if __name__ == '__main__':
    run()

