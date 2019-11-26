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

import astropy.units as u 
time_conv = (1 * u.kpc/(u.km/u.s)).to_value(u.Myr)

sims_list = ['/galakos/lvl5', '/galakos/lvl4', '/galakos/lvl3-hernquist']
sims_list = [base+s for s in sims_list]
name_list = ['lvl5']#, 'lvl4']#, 'lvl3']

Rbin = 5
firstkey = 250

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

for sim, name in zip(sims_list, name_list):
    out = {}
    out['firstkey'] = firstkey
    
    indices, files = mwib_io.get_files_indices(sim+'/output/*.hdf5')
    fourier = h5.File(sim+'/analysis/fourier_components.hdf5', mode='r')

    Rlist = np.array([np.array(fourier['snapshot_'+"{0:03}".format(idx)]['Rlist']) for idx in tqdm(indices)])
    A2r = np.array([np.array(fourier['snapshot_'+"{0:03}".format(idx)]['A2r']) for idx in tqdm(indices)])
    A2i = np.array([np.array(fourier['snapshot_'+"{0:03}".format(idx)]['A2i']) for idx in tqdm(indices)])

    time_list = np.array(fourier['time'])
    out['time'] = time_list
    out['firsttime'] = time_list[firstkey]

    phi = np.arctan2(A2i, A2r)
    phi = phi[:,Rbin]

    A2mag = np.sqrt(np.add(np.square(A2i), np.square(A2r)))
    A2mag = A2mag[:,Rbin]

    out['phi'] = phi
    out['RatRbin'] = Rlist[:,Rbin]
    out['A2mag'] = A2mag

    bar_angle = compute_bar_angle(phi, firstkey=firstkey)

    out['bar_angle'] = bar_angle

    pfit = [np.polyfit(time_list[firstkey:], bar_angle[firstkey:], i) for i in range(10)]
    out['poly_fit'] = pfit

    pickle.dump(out, open('bar_angle_'+name+'.p', 'wb'))

    # pickle.dump((star_birthtime, star_mini), open('SFR_'+name+'.p', 'wb'))
    
