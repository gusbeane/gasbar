import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from astropy.io import fits
from scipy.interpolate import interp1d
import re
import pickle
import agama

import astropy.units as u
from scipy.optimize import minimize

agama.setUnits(mass=1E10, length=1, velocity=1)

from photutils.isophote import EllipseGeometry, Ellipse

from matplotlib import rc
mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

basepath = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/'
snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = basepath + '/analysis/bar_prop/data/'

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'
# phS2R35 = 'smuggle'

lvl = 'lvl3'

range_xy = [[-10, 10], [-10, 10]]
nres = 256
extent = [range_xy[0][0], range_xy[0][1], range_xy[1][0], range_xy[1][1]]
dx = (range_xy[0][1] - range_xy[0][0])/nres
dy = (range_xy[1][1] - range_xy[1][0])/nres

vmin=1E-3
vmax=10.**(0.5)

def read_snap(idx, name, lvl, parttype=[1, 2, 3, 4], fields=['Coordinates', 'Masses', 'Potential']):
    sn = arepo.Snapshot(snap_path+name+'/'+lvl+'/output', idx, combineFiles=True, parttype=parttype,
                        fields=fields)
    return sn

def read_fourier(name, lvl='lvl3', 
                 basepath='/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/'):
    f = h5.File(basepath+'/fourier_component/data/fourier_'+name+'-'+lvl+'.hdf5', mode='r')
    return f

def read_agama_pot(idx, name, lvl):
    base = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/agama_pot/data/'
    fname = base + 'pot_' + name + '-' + lvl + '/pot_' + name + '-' + lvl + '.' + str(idx) + '.txt'
    return agama.Potential(fname)

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

def rotate_pos(pos, ang):

    Rmat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang),  np.cos(ang), 0.0],
                     [0.0,         0.0,          1.0]])
    
    pos = np.swapaxes(pos, 0, 1)
    pos = np.matmul(Rmat, pos)
    pos = np.swapaxes(pos, 0, 1)
    
    return pos

def get_bar_angle(phi, firstkey):
    out = np.zeros(len(phi))

    # set the first bar anglea
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

def get_sorted_keys(dat):
    keys = list(dat.keys())
    # only keep keys that are snapshot keys
    keys = [k for k in keys if 'snapshot' in k]

    # extract and sort indices
    indices = [int(re.findall(r'\d?\d?\d\d\d', k)[0]) for k in keys]
    sorted_arg = np.argsort(indices)
    keys_sorted = [keys[i] for i in sorted_arg]

    return keys_sorted

def get_A2_angle(dat, keys, Rbin, cum=False):
    if cum:
        rk = 'A2r_c'
        ri = 'A2i_c'
    else:
        rk = 'A2r'
        ri = 'A2i'
    
    Rlist = np.array([np.array(dat[k]['Rlist']) for k in keys])
    A2r = np.array([np.array(dat[k][rk]) for k in keys])
    A2i = np.array([np.array(dat[k][ri]) for k in keys])
    phi = np.arctan2(A2i, A2r)
    phi = phi[:,Rbin]
    R_at_Rbin = Rlist[:,Rbin]
    
    time = np.array(dat['time'])
    time = time * u.Myr
    time = time.to_value(u.kpc/(u.km/u.s))

    return time, R_at_Rbin, phi

def main_bar_angle(dat, Rbin = 3, firstkey = 150, nmax = 10, cum=False):
    # try loading snapshot
#     dat = h5.File(fname, mode='r')
    out = {}

    keys = get_sorted_keys(dat)
    time, R, phi = get_A2_angle(dat, keys, Rbin, cum=cum)
    bar_angle = get_bar_angle(phi, firstkey)

    pattern_speed = np.gradient(bar_angle, time)

    out['time'] = time
    out['firstkey'] = firstkey
    out['R'] = R
    out['phi'] = phi
    out['bar_angle'] = bar_angle
    out['pattern_speed'] = pattern_speed

    return out

def get_pos_mass(sn, bangle):
    pos = []
    mass = []
    
    center = sn.part1.pos.value[np.argmin(sn.part1.pot)]
    
    for pt in [2, 3, 4]:
        if sn.NumPart_Total[pt] > 0:
            part = getattr(sn, 'part'+str(pt))
            this_pos = part.pos.value - center
            this_pos = rotate_pos(this_pos, -bangle)
            pos.append(this_pos)
            
            if sn.MassTable[pt] > 0.0:
                mass.append( np.full(sn.NumPart_Total[pt], sn.MassTable[pt].value) )
            else:
                mass.append( part.mass.value )
    
    pos = np.concatenate(pos)
    mass = np.concatenate(mass)
    
    return pos, mass

def get_heatmap(sn, bangle):
    pos, mass = get_pos_mass(sn, bangle)
    
    dx = (range_xy[0][1] - range_xy[0][0])/nres
    dy = (range_xy[1][1] - range_xy[1][0])/nres
    surf = dx * dy
    
    heatmap, _, _ = np.histogram2d(pos[:,0], pos[:,1], 
                                   bins=(nres, nres), range=range_xy, weights=mass/surf)
    
    return heatmap

def get_bar_length(isolist):
    k = np.argmax(isolist.eps)
    ellip = np.max(isolist.eps)
    Rbar_e = isolist.sma[k] * dx
    
    kpa = k
    for i in range(len(isolist)-k):
        kpa += 1
        delta_PA = np.abs(isolist[kpa].pa - isolist[k].pa) * 180/np.pi
        # print(delta_PA)
        if delta_PA > 5:
            kpa -= 1
            break
    
    Rbar_PA = isolist.sma[kpa] * dx
    
    return Rbar_e, Rbar_PA, ellip, k, kpa

def fit_ellipse(heatmap):
    # geometry = EllipseGeometry(x0=0.0, y0=0.0, sma=80, eps=0.5, pa=0.0)
    ellipse = Ellipse(heatmap.T)
    
    isolist = ellipse.fit_image(step=4.0, minsma=10., maxsma=200., linear=True)
    
    Rbar_e, Rbar_PA, ellip, ke, kpa = get_bar_length(isolist)
    
    return isolist, ke, kpa
    
def run():
    nres = 256
    rng = [[-10., 10.], [-10., 10.]]

    Nbody_idx = [500, 700, 900]
    SMUGGLE_idx = [200, 400, 600]

    name_list = [Nbody, phS2R35]

    extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]

    cm = 1/2.54
    
    fourierN = read_fourier(Nbody)
    fourierS = read_fourier(phS2R35)
    
    bangleN = main_bar_angle(fourierN)
    bangleS = main_bar_angle(fourierS)
    
    barpropN = read_bar_prop(Nbody, 'lvl3')
    barpropS = read_bar_prop(phS2R35, 'lvl3')

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(textwidth*cm, (12/16)*textwidth*cm))

    i = 0
    for Nidx, Sidx in zip(Nbody_idx, SMUGGLE_idx):
        snN = read_snap(Nidx, Nbody, 'lvl3')
        snS = read_snap(Sidx, phS2R35, 'lvl3')
        
        agama_potN = read_agama_pot(Nidx, Nbody, 'lvl3')
        agama_potS = read_agama_pot(Sidx, phS2R35, 'lvl3')
        
        print('N ps=', bangleN['pattern_speed'][Nidx])
        print('S ps=', bangleS['pattern_speed'][Nidx])
        
        RCR_N = compute_RCR(agama_potN, bangleN['pattern_speed'][Nidx])
        RCR_S = compute_RCR(agama_potS, bangleS['pattern_speed'][Nidx])
        
        print('RCR_N=', RCR_N)
        print('RCR_S=', RCR_S)
        
        heatmapN = get_heatmap(snN, bangleN['bar_angle'][Nidx])
        heatmapS = get_heatmap(snS, bangleS['bar_angle'][Sidx])
        
        ax[0][i].imshow(heatmapN.T, extent=extent, origin='lower', 
                        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        
        ax[1][i].imshow(heatmapS.T, extent=extent, origin='lower', 
                        norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
        
        RbN = barpropN['Rbar'][Nidx]
        ax[0][i].plot([-RbN, RbN], [0, 0], c='w')
        
        RbS = barpropS['Rbar'][Nidx]
        ax[1][i].plot([-RbS, RbS], [0, 0], c='w')

        # print('Nbody', Nidx)
        # try:
        #     isolist, ke, kpa = pickle.load(open('Nbody'+str(Nidx)+'.p', 'rb'))
        # except:
        #     isolist, ke, kpa = fit_ellipse(heatmapN)
        #     pickle.dump((isolist, ke, kpa), open('Nbody'+str(Nidx)+'.p', 'wb'))
        # sma = isolist.sma[kpa]
        # iso = isolist.get_closest(sma)
        # x, y = iso.sampled_coordinates()
        # x = (x-nres/2) * dx
        # y = (y-nres/2) * dy
        # ax[0][i].plot(x, y, color='white')
        
        # print('Rot param N', Nidx, RCR_N/sma/dx)
        
        # print('SMUGGLE', Sidx)
        # try:
        #     isolist, ke, kpa = pickle.load(open('SMUGGLE'+str(Sidx)+'.p', 'rb'))
        # except:
        #     isolist, ke, kpa = fit_ellipse(heatmapS)
        #     pickle.dump((isolist, ke, kpa), open('SMUGGLE'+str(Sidx)+'.p', 'wb'))
        # sma = isolist.sma[kpa]
        # iso = isolist.get_closest(sma)
        # x, y = iso.sampled_coordinates()
        # x = (x-nres/2) * dx
        # y = (y-nres/2) * dy
        # ax[1][i].plot(x, y, color='white')
        
        # print('Rot param S', Sidx, RCR_S/sma/dx)
        
        # plot CR as a circle
        xlist = np.linspace(-RCR_N, RCR_N, 1000)
        ylist = np.sqrt(RCR_N**2-xlist**2)
        ax[0][i].plot(xlist, ylist, c='w', ls='dashed')
        ax[0][i].plot(xlist, -ylist, c='w', ls='dashed')
        
        xlist = np.linspace(-RCR_S, RCR_S, 1000)
        ylist = np.sqrt(RCR_S**2-xlist**2)
        ax[1][i].plot(xlist, ylist, c='w', ls='dashed')
        ax[1][i].plot(xlist, -ylist, c='w', ls='dashed')
        
        i += 1
    
    ax[0][0].set_title(r'$t=1\,\textrm{Gyr}$')
    ax[0][1].set_title(r'$t=2\,\textrm{Gyr}$')
    ax[0][2].set_title(r'$t=3\,\textrm{Gyr}$')

    ax[0][0].set_ylabel(r'without interstellar medium')
    ax[1][0].set_ylabel(r'with interstellar medium')
    
    for x in ax.ravel():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
        
    ax[1][2].plot([6.5, 8.5], [-8, -8], c='w', lw=2)
    ax[1][2].text(7.5, -7.5, r'$2\,\textrm{kpc}$', c='w', ha='center')

    fig.tight_layout()

    fig.savefig('fig1.pdf')
    # fig.savefig('mockHST.png', dpi=256)

if __name__ == '__main__':
    run()

