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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
fourier_path = '/n/home01/abeane/starbar/analysis/fourier_sphere/data/'
bprop_path = '/n/home01/abeane/starbar/analysis/bar_prop/data/'

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

@njit
def compute_surface_density(R, mass, Rbins):
    key = np.where(R < Rbins[-1])[0]
    R = R[key]
    mass = mass[key]
    
    surf_dens = np.zeros(len(Rbins)-1)
    ave_R = np.zeros(len(Rbins)-1)
    N_in_bin = np.zeros(len(Rbins)-1)
    
    for i in range(len(R)):
        for j in range(len(Rbins)-1):
            if R[i] >= Rbins[j] and R[i] < Rbins[j+1]:
                ave_R[j] += R[i]
                N_in_bin[j] += 1
                surf_dens[j] += mass[i]
    
    for j in range(len(Rbins)-1):
        if N_in_bin[j] > 0:
            ave_R[j] /= N_in_bin[j]
            surf_dens[j] /= np.pi * (Rbins[j+1]**2 - Rbins[j]**2)
        else:
            ave_R[j] = np.nan
            surf_dens[j] = np.nan
    
    return ave_R, surf_dens

def rotate_pos(pos, ang):

    Rmat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang),  np.cos(ang), 0.0],
                     [0.0,         0.0,          1.0]])
    
    pos = np.swapaxes(pos, 0, 1)
    pos = np.matmul(Rmat, pos)
    pos = np.swapaxes(pos, 0, 1)
    
    return pos

def read_fourier(name, lvl):
    t = h5.File(fourier_path + 'fourier_' + name + '-' + lvl + '.hdf5', mode='r')
    out = {}
    for key in t.keys():
        out[key] = t[key][:]
    
    out['A2_angle'] = np.arctan2(out['A2i'], out['A2r'])
    out['A2_h_angle'] = np.arctan2(out['A2i_h'], out['A2r_h'])
    
    return out

def read_bar_prop(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = {}
    for key in t['bar_prop'].keys():
        out[key] = t['bar_prop'][key][:]
    
    out['bar_angle'] = t['bar_angle'][:]
    t.close()
    
    return out

def compute_dphi(phi_disk, phi_halo):
    dphi = phi_disk - phi_halo
    dphi = np.mod(dphi + np.pi, 2.*np.pi) - np.pi 
    dphi /= 2.0
    dphi = savgol_filter(dphi, 81, 3)

    return dphi

def run():
    cm = 1/2.54
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(9*cm, 14*cm))
    idx_list = [0, 100, 200, 400, 600, 800]
    center = np.array([200., 200., 200.])

    data_mol = np.genfromtxt('sigmol-G1a-4p5.txt')
    data_sfr = np.genfromtxt('sigsfr-G1a-4p5.txt')
    data_atomic = np.genfromtxt('surf_all.out')

    for i,idx in enumerate(tqdm(idx_list)):
        sn = read_snap(idx, phS2R35, lvl, parttype=[0], fields=None)

        pos = sn.part0.pos.value - center
        R = np.linalg.norm(pos[:,:2], axis=1)
        Rbins = np.logspace(-2, np.log10(20), 25)

        mass = sn.part0.mass.value * sn.part0.MolecularHFrac
        mass_atomic = sn.part0.mass.value * (1-sn.part0.MolecularHFrac)
        if i==0:
            r = np.linalg.norm(pos, axis=1)

        aveR, surf = compute_surface_density(R, mass, Rbins)
        aveR_atomic, surf_atomic = compute_surface_density(R, mass_atomic, Rbins)

        if True:
            label_0 = repr(sn.Time.value)
            label_1 = None
        else:
            label_0 = None
            label_1 = repr(sn.Time.value)

        ax[0].plot(aveR_atomic, (1E10/1E6) * surf_atomic, label=label_0, c=tb_c[i])
        ax[1].plot(aveR, (1E10 / 1E6) * surf, label=label_1, c=tb_c[i]) # convert to Msun/pc^2

        sfr = sn.part0.sfr.value
        key = np.where(sfr > 0)[0]
        Rbins = np.logspace(-2, np.log10(20), 15)
        aveR, surf = compute_surface_density(R[key], sfr[key], Rbins)

        surf[np.logical_or(np.isnan(surf), surf==0.0)] = 1E-9
        print(surf)

        ax[2].plot(aveR, (1E9 / 1E6) * surf, label=label_1, c=tb_c[i]) # convert to Msun/Gyr/pc^2

    ax[0].plot(data_atomic[:,0], 0.008*data_atomic[:,3], c='k', ls='dashed') # convert to Msun/pc^2 from 1E18 /cm^2
    ax[1].plot(data_mol[:,0], (1./1E6) * data_mol[:,1], c='k', ls='dotted') # convert to Msun/pc^2
    ax[2].plot(data_sfr[:,0], (1E9/1E6) * data_sfr[:,1], c='k', ls='dotted') # convert to Msun/Gyr/pc^2

    custom_lines = [mpl.lines.Line2D([0], [0], color='k', ls='dashed'),
                    mpl.lines.Line2D([0], [0], color='k', ls='dotted')]
    ax[1].legend(custom_lines, ['Kalberla \& Dedes (2008)', 'Evans et al. (2022)'], frameon=False)

    # ax[0].axhline(10, c='k', ls='dashed')
    # ax[1].axhline(10.**(0.5), c='k', ls='dashed')

    ax[0].set(yscale='log')
    ax[1].set(yscale='log')
    ax[2].set(yscale='log')
    ax[2].set(xlim=(0, 10))

    ax[1].set_ylim(1E-1, 1E2)
    ax[2].set_ylim(1E-1, 1E2)
    ax[0].set_ylim(1E-1, 1E2)

    ax[0].set(ylabel=r'$\Sigma_{\textrm{atomic}}\,[\,M_{\odot} / \textrm{pc}^2\,]$')
    ax[1].set(ylabel=r'$\Sigma_{\textrm{mol}}\,[\,M_{\odot} / \textrm{pc}^2\,]$')
    ax[2].set(ylabel=r'$\Sigma_{\textrm{SFR}}\,[\,M_{\odot} / \textrm{Gyr} / \textrm{pc}^2\,]$')
    ax[2].set(xlabel=r'$R\,[\,\textrm{kpc}\,]$')

    ax[0].legend(frameon=False, title=r'$t\,[\,\text{Gyr}\,]$', ncol=2)
    # ax[1].legend(frameon=False, title=r'$t\,[\,\text{Gyr}\,]$', ncol=2)
    ax[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig('surf_dens.pdf')


if __name__ == '__main__':
    run()
