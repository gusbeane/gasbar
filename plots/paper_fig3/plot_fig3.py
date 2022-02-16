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

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
fourier_path = '/n/home01/abeane/starbar/plots/fourier_sphere/data/'
bprop_path = '/n/home01/abeane/starbar/plots/bar_prop/data/'

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
    
    surf_dens = np.zeros(len(Rbins)-1)
    ave_R = np.zeros(len(Rbins)-1)
    N_in_bin = np.zeros(len(Rbins)-1)
    
    for i in range(len(R)):
        for j in range(len(Rbins)-1):
            if R[i] >= Rbins[j] and R[i] < Rbins[j+1]:
                ave_R[j] += R[i]
                N_in_bin[j] += 1
                surf_dens[j] += mass
    
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

def compute_heatmap(sn, center, nres, rng, angle=0.0):
    pos = sn.part1.pos.value - center
    mass = sn.MassTable[1]

    pos = rotate_pos(pos, angle)

    Rbins = np.logspace(-3, 2, 80)
    R = np.linalg.norm(pos[:,:2], axis=1)
    ave_R, surf = compute_surface_density(R, mass, Rbins)

    surf_interp = interp1d(ave_R, surf, fill_value='extrapolate')

    heatmap, xbins, ybins = np.histogram2d(pos[:,0], pos[:,1], bins=(nres, nres), range=rng)
    heatmap *= mass

    dx = (rng[0][1] - rng[0][0])/nres
    dy = (rng[1][1] - rng[1][0])/nres
    heatmap /= dx * dy

    ave_x = (xbins[:-1] + xbins[1:])/2.0
    ave_y = (ybins[:-1] + ybins[1:])/2.0
    xgrid, ygrid = np.meshgrid(ave_x, ave_y, indexing='ij')
    Rgrid = np.sqrt(xgrid * xgrid + ygrid * ygrid)
    surf_grid = surf_interp(Rgrid)

    heatmap -= surf_grid
    heatmap = gaussian_filter(heatmap, sigma=8)

    return heatmap

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
    fourier_Nbody = read_fourier(Nbody, lvl)
    fourier_SMUGGLE = read_fourier(phS2R35, lvl)

    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)

    Nbody_idx = 820
    SMUGGLE_idx = 520

    nres = 256
    rng = [[-7.5, 7.5], [-7.5, 7.5]]
    extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]
    
    vmin = -0.005
    vmax = 0.005

    fig, ax = plt.subplots(1, 3, figsize=(8, 2.5))

    # First panel, wake of Nbody.
    sn = read_snap(Nbody_idx, Nbody, lvl, parttype=[1])
    heatmap = compute_heatmap(sn, np.array([0., 0., 0.]), nres, rng, angle=-fourier_Nbody['A2_angle'][Nbody_idx]/2.0)
    # heatmap = compute_heatmap(sn, np.array([0., 0., 0.]), nres, rng, angle=-fourier_Nbody['A2_angle'][Nbody_idx])
    Rbar = bar_prop_Nbody['Rbar'][Nbody_idx]
    print(fourier_Nbody['A2_angle'][Nbody_idx], bar_prop_Nbody['bar_angle'][Nbody_idx], fourier_Nbody['A2_h_angle'][Nbody_idx])

    ax[0].imshow(heatmap.T, origin='lower', vmin=vmin, vmax=vmax, extent=extent, cmap='bwr')
    ax[0].plot((-Rbar, Rbar), (0.0, 0.0), c='k')
    dphi = fourier_Nbody['A2_h_angle'][Nbody_idx] - fourier_Nbody['A2_angle'][Nbody_idx]
    dphi /= 2.0
    ax[0].plot((-Rbar*np.cos(dphi), Rbar*np.cos(dphi)), (-Rbar*np.sin(dphi), Rbar*np.sin(dphi)), c='k', ls='dashed')

    ax[0].set_aspect('auto')

    # Second panel, diff in angle.
    dphiN = compute_dphi(fourier_Nbody['A2_angle'], fourier_Nbody['A2_h_angle'])
    dphiS = compute_dphi(fourier_SMUGGLE['A2_angle'], fourier_SMUGGLE['A2_h_angle'])

    tN = fourier_Nbody['time']
    tS = fourier_SMUGGLE['time']

    ax[1].plot(tN - tN[300], 180.*dphiN, c=tb_c[0])
    ax[1].plot(tS, 180.*dphiS, c=tb_c[1])
    ax[1].axhline(0.0, c='k')

    ax[1].axvline(tS[520])

    ax[1].set(xlim=(0, 5), ylim=(-35, 70), xlabel=r'$t\,[\,\text{Gyr}\,]$', ylabel=r'$\text{angle difference}\,[\,\text{deg}\,]$')

    # Third panel, wake of SMUGGLE.
    sn = read_snap(SMUGGLE_idx, phS2R35, lvl, parttype=[1])
    heatmap = compute_heatmap(sn, np.array([200., 200., 200.]), nres, rng, angle=-fourier_SMUGGLE['A2_angle'][SMUGGLE_idx]/2.0)
    # heatmap = compute_heatmap(sn, np.array([0., 0., 0.]), nres, rng, angle=-fourier_SMUGGLE['A2_angle'][SMUGGLE_idx])
    Rbar = bar_prop_SMUGGLE['Rbar'][SMUGGLE_idx]
    print(fourier_SMUGGLE['A2_angle'][SMUGGLE_idx], bar_prop_SMUGGLE['bar_angle'][SMUGGLE_idx], fourier_SMUGGLE['A2_h_angle'][SMUGGLE_idx])

    ax[2].imshow(heatmap.T, origin='lower', vmin=vmin, vmax=vmax, extent=extent, cmap='bwr')
    ax[2].plot((-Rbar, Rbar), (0.0, 0.0), c='k')
    dphi = fourier_SMUGGLE['A2_h_angle'][SMUGGLE_idx] - fourier_SMUGGLE['A2_angle'][SMUGGLE_idx]
    dphi /= 2.0
    ax[2].plot((-Rbar*np.cos(dphi), Rbar*np.cos(dphi)), (-Rbar*np.sin(dphi), Rbar*np.sin(dphi)), c='k', ls='dashed')

    ax[2].set_aspect('auto')


    fig.tight_layout()

    fig.savefig('fig3.pdf')

if __name__ == '__main__':
    run()
