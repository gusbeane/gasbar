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
import re
import astropy.units as u
import warnings

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

def read_fourier(name, lvl='lvl3', 
                 basepath='/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/'):
    f = h5.File(basepath+'/fourier_component/data/fourier_'+name+'-'+lvl+'.hdf5', mode='r')
    return f

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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
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

    return time, R_at_Rbin, phi

def get_bar_length(dat, keys, Rmin=2, Rmax=10, ratio_cut = 2):
    Rlist = np.array([np.array(dat[k]['Rlist']) for k in keys])

    A0 = np.array([np.array(dat[k]['A0']) for k in keys])
    A1r = np.array([np.array(dat[k]['A1r']) for k in keys])
    A1i = np.array([np.array(dat[k]['A1i']) for k in keys])
    A2r = np.array([np.array(dat[k]['A2r']) for k in keys])
    A2i = np.array([np.array(dat[k]['A2i']) for k in keys])
    A3r = np.array([np.array(dat[k]['A3r']) for k in keys])
    A3i = np.array([np.array(dat[k]['A3i']) for k in keys])
    A4r = np.array([np.array(dat[k]['A4r']) for k in keys])
    A4i = np.array([np.array(dat[k]['A4i']) for k in keys])
    A5r = np.array([np.array(dat[k]['A5r']) for k in keys])
    A5i = np.array([np.array(dat[k]['A5i']) for k in keys])
    A6r = np.array([np.array(dat[k]['A6r']) for k in keys])
    A6i = np.array([np.array(dat[k]['A6i']) for k in keys])
    
    I0 = A0/2.
    I1 = np.sqrt(A1r*A1r + A1i*A1i)
    I2 = np.sqrt(A2r*A2r + A2i*A2i)
    I3 = np.sqrt(A3r*A3r + A3i*A3i)
    I4 = np.sqrt(A4r*A4r + A4i*A4i)
    I5 = np.sqrt(A5r*A5r + A5i*A5i)
    I6 = np.sqrt(A6r*A6r + A6i*A6i)
    
    Ib = I0 + I2 + I4 + I6
    Iib = I0 - I2 + I4 - I6
    
    IbIib = Ib/Iib
    
    Rbar_list = []
    for i,k in enumerate(keys):
        R = Rlist[i,:]
        ratio = IbIib[i,:]
        
        Rkey = np.logical_and(R > Rmin, R< Rmax)
        ratio = ratio[Rkey]
        R = R[Rkey]
        j = 0
        try:
            while ratio[j] > ratio_cut:
                j += 1
            Rbar = R[j-1] + (ratio_cut - ratio[j-1]) * (R[j]-R[j-1])/(ratio[j]-ratio[j-1])
        except:
            Rbar = np.nan
        Rbar_list.append(Rbar)

    time = np.array(dat['time'])    
    
    return time, np.array(Rbar_list)

def evaluate_polynomial(pfit, n, time, bar_angle_firstkey, firstkey):
    pfit_n = pfit[n]
    poly_bar_angle = np.zeros(len(time))
    poly_pattern_speed = np.zeros(len(time))

    for i in range(n+1):
        ba = pfit_n[i] * time ** (n-i)
        poly_bar_angle[firstkey:] += ba[firstkey:]
        ps = (n-i) * pfit_n[i] * time**(n-1-i)
        poly_pattern_speed[firstkey:] += ps[firstkey:]

    poly_bar_angle[:firstkey] += bar_angle_firstkey

    poly_pattern_speed = poly_pattern_speed / u.Myr
    poly_pattern_speed = poly_pattern_speed.to_value(u.km/u.s/u.kpc)

    return poly_bar_angle, poly_pattern_speed

def main_bar_angle(dat, Rbin = 3, firstkey = 150, nmax = 10, cum=False):
    # try loading snapshot
#     dat = h5.File(fname, mode='r')
    out = {}

    keys = get_sorted_keys(dat)
    time, R, phi = get_A2_angle(dat, keys, Rbin, cum=cum)
    time, Rbar = get_bar_length(dat, keys)
#     Rlist, Iibar = get_bar_length(dat, keys)
    bar_angle = get_bar_angle(phi, firstkey)

    pattern_speed = np.gradient(bar_angle, time) / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    pfit = [np.polyfit(time[firstkey:], bar_angle[firstkey:], i) for i in range(nmax)]
    
    out['time'] = time
    out['firstkey'] = firstkey
    out['R'] = R
    out['Rbar'] = Rbar
    out['phi'] = phi
    out['bar_angle'] = bar_angle
    out['pattern_speed'] = pattern_speed
    out['pfit'] = pfit

    # now evaluate the polynomial for each fit and save the result
    out['poly_eval'] = {}
    for n in range(nmax):
        poly_bar_angle, poly_pattern_speed = evaluate_polynomial(pfit, n, time, bar_angle[firstkey], firstkey)

        out['poly_eval'][n] = (poly_bar_angle, poly_pattern_speed)

    return out
#     return Rlist, Iibar

def run():
    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)

    print(bar_prop_Nbody['pattern_speed'][0:300])
    
    agama_pot_Nbody = read_all_agama_pot(Nbody, lvl)
    agama_pot_SMUGGLE = read_all_agama_pot(phS2R35, lvl)
    
    fourierN = read_fourier(Nbody)
    fourierS = read_fourier(phS2R35)
    
    bar_outN = main_bar_angle(fourierN)
    bar_outS = main_bar_angle(fourierS)
    
    psN = savgol_filter(bar_outN['pattern_speed'], 81, 3)
    psS = savgol_filter(bar_outS['pattern_speed'], 81, 3)

    RCR_Nbody = []
    for i in range(len(bar_prop_Nbody['tlist'])):
        # if bar_prop_Nbody['pattern_speed'][i] <= 0.0:
        #     RCR_Nbody.append(np.nan)
        # else:
        #     RCR = compute_RCR(agama_pot_Nbody[i], bar_prop_Nbody['pattern_speed'][i])
        #     RCR_Nbody.append(RCR)
        if psN[i] <= 0.0:
            RCR_Nbody.append(np.nan)
        else:
            RCR = compute_RCR(agama_pot_Nbody[i], psN[i])
            RCR_Nbody.append(RCR)
    
    RCR_SMUGGLE = []
    for i in range(len(bar_prop_SMUGGLE['tlist'])):
        RCR = compute_RCR(agama_pot_SMUGGLE[i], psS[i])
        RCR_SMUGGLE.append(RCR)

    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(9*cm, 9*cm))

    

    # Second panel, length of bar and mass of bar.
    t300 = bar_prop_Nbody['tlist'][300]

    rot_Nbody = np.array(RCR_Nbody)/bar_outN['Rbar']
    rot_SMUGGLE = np.array(RCR_SMUGGLE) / bar_outS['Rbar']

    rot_Nbody[rot_Nbody==np.nan] = 1.4
    rot_SMUGGLE[rot_SMUGGLE==np.nan] = 1.4
    
    # rot_Nbody = savgol_filter(rot_Nbody, 81, 3)
    # rot_SMUGGLE = savgol_filter(rot_SMUGGLE, 81, 3)

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

