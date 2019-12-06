import sys
mach = sys.platform
if mach == 'linux':
    sys.path.append('/n/home01/abeane/mwib_analysis')
    base = '/n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo'
elif mach == 'darwin':
    sys.path.append('/Users/abeane/scratch/mwib_analysis')
    base = '/Users/abeane/scratch/mwib_runs/arepo'

from mwib_analysis import mwib_io
import numpy as np
import h5py as h5
from tqdm import tqdm
import pickle
import re

import astropy.units as u 
time_conv = (1 * u.kpc/(u.km/u.s)).to_value(u.Myr)

sims_list = ['/galakos/lvl5', '/galakos/lvl4', '/galakos/lvl3-hernquist',
             '/galakos-rotbulge/lvl5', '/galakos-rotbulge/lvl4',
             '/galakos-softtest/lvl5-fac2', '/galakos-softtest/lvl5-fac4', '/galakos-softtest/lvl5-fac8',
             '/galakos-softtest/lvl4-fac2', '/galakos-softtest/lvl4-fac4', '/galakos-softtest/lvl4-fac8']
sims_list = [base+s for s in sims_list]
name_list = ['lvl5', 'lvl4', 'lvl3', 'lvl5-rotbulge', 'lvl4-rotbulge',
             'lvl5-fac2', 'lvl5-fac4', 'lvl5-fac8', 'lvl4-fac2', 'lvl4-fac4', 'lvl4-fac8']

Rbin = 5
Rbin_max = 10
firstkey_list = [250, 250, 250, 
                 250, 250, 
                 250, 250, 250,
                 50, 50, 50]

def compute_bar_angle(phi, firstkey=400):
    out = np.zeros(len(phi))

    # set the first bar angle
    first_bar_angle = phi[firstkey]/2.0
    out[firstkey] = first_bar_angle
    
    # set all subsequent angles
    for i in np.arange(firstkey+1, len(out)):
        dphi = phi[i] - phi[i-1]
        if dphi < 0:
            dphi += 2.*np.pi
        out[i] = out[i-1] + dphi/2.0

    # set all previous angles to be the bar angle
    for i in np.arange(0, firstkey):
        out[i] = first_bar_angle

    return out

for sim, name, firstkey in zip(sims_list, name_list, firstkey_list):
    out = {}
    out['firstkey'] = firstkey
    
    #indices, files = mwib_io.get_files_indices(sim+'/output/*.hdf5')
    fourier = h5.File(sim+'/analysis/fourier_components.hdf5', mode='r')

    keys = list(fourier.keys())
    keys = [k for k in keys if 'snapshot' in k]
    indices = [int(re.findall(r'\d?\d?\d\d\d', k)[0]) for k in keys]
    sorted_arg = np.argsort(indices)
    keys_sorted = [keys[i] for i in sorted_arg]

    Rlist = np.array([np.array(fourier[k]['Rlist']) for k in tqdm(keys_sorted)])
    A2r = np.array([np.array(fourier[k]['A2r']) for k in tqdm(keys_sorted)])
    A2i = np.array([np.array(fourier[k]['A2i']) for k in tqdm(keys_sorted)])
    A0 = np.array([np.array(fourier[k]['A0']) for k in tqdm(keys_sorted)])

    time_list = np.array(fourier['time'])
    out['time'] = time_list
    out['firsttime'] = time_list[firstkey]

    phi = np.arctan2(A2i, A2r)
    phi = phi[:,Rbin]

    A2mag = np.sqrt(np.add(np.square(A2i), np.square(A2r)))
    A2mag_at_R = A2mag[:,Rbin]

    A2A0 = np.divide(A2mag, A0)
    A2A0_max = np.max(A2A0[:,:Rbin_max], axis=1)

    out['phi'] = phi
    out['RatRbin'] = Rlist[:,Rbin]
    out['RatRbin_max'] = Rlist[:,Rbin_max]

    out['A2mag'] = A2mag_at_R
    out['A2A0_max'] = A2A0_max

    bar_angle = compute_bar_angle(phi, firstkey=firstkey)

    out['bar_angle'] = bar_angle

    pfit = [np.polyfit(time_list[firstkey:], bar_angle[firstkey:], i) for i in range(10)]
    out['poly_fit'] = pfit

    pickle.dump(out, open('bar_angle_'+name+'.p', 'wb'))

    # pickle.dump((star_birthtime, star_mini), open('SFR_'+name+'.p', 'wb'))
    
