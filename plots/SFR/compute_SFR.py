import sys
mach = sys.platform
if mach == 'linux':
    sys.path.append('/n/home01/abeane/mwib_analysis')
    base = '/n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo'
elif mach == 'darwin':
    sys.path.append('/Users/abeane/scratch/mwib_analysis')
    base = '/Volumes/abeaneSSD001/mwib_runs/arepo'

print(base)

from mwib_analysis import mwib_io
import numpy as np
import h5py as h5
from tqdm import tqdm
import pickle

import astropy.units as u 
time_conv = (1 * u.kpc/(u.km/u.s)).to_value(u.Myr)

sims_list = ['/galakos/lvl5', '/galakos/lvl4', '/galakos/lvl3-hernquist']
sims_list = [base+s for s in sims_list]
name_list = ['lvl5', 'lvl4', 'lvl3']

skip = 10

for sim, name in zip(sims_list, name_list):
    indices, files = mwib_io.get_files_indices(sim+'/output/*.hdf5')

    print(len(files))
    print(sim+'/output/*.hdf5')

    f = files[-1]
    t = h5.File(f, mode='r')
    star_birthtime = np.array(t['PartType4']['GFM_StellarFormationTime'])*time_conv
    star_mini = np.array(t['PartType4']['GFM_InitialMass'])*1E10

    pickle.dump((star_birthtime, star_mini), open('SFR_'+name+'.p', 'wb'))
    
