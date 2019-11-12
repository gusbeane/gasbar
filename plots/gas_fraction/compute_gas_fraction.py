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

sims_list = ['/galakos/lvl5', '/galakos/lvl4', '/galakos/lvl3']
sims_list = [base+s for s in sims_list]
name_list = ['lvl5', 'lvl4', 'lvl3']

skip = 10

for sim, name in zip(sims_list, name_list):
    indices, files = mwib_io.get_files_indices(sim+'/output/*.hdf5')
    gas_fraction_list = []
    time_list = []
    for idx, f in zip(tqdm(indices[::skip]), files[::skip]):
        time = mwib_io.get_time(f)
        # dont need to subtract COM since not necessary
        snap = mwib_io.read_snap(f, subtract_com=False)

        gas_mass = np.sum(snap['PartType0']['Masses'])
        disk_mass = np.sum(snap['PartType2']['Masses'])
        try:
            star_mass = np.sum(snap['PartType4']['Masses'])
        except KeyError:
            star_mass = 0.0
        gas_fraction = gas_mass / (gas_mass + disk_mass + star_mass)

        gas_fraction_list.append(gas_fraction)
        time_list.append(time)

    time_list = np.array(time_list)
    gas_fraction_list = np.array(gas_fraction_list)
    pickle.dump((time_list, gas_fraction_list), open('gas_fraction_'+name+'.p', 'wb'))
    
