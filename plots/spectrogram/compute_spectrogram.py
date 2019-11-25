# import numpy as np
# import sys
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import astropy.units as u

# import matplotlib as mpl

# sys.path.append('/Users/abeane/scratch/mwib_analysis')
# from mwib_analysis import mwib_io, mwib_agama, disk

# path_list = ['/Volumes/abeaneSSD001/mwib_runs/arepo/galakos/lvl4',
#              '/Volumes/abeaneSSD001/mwib_runs/arepo/galakos-dry/lvl4']
# name_list = ['SMUGGLE', 'Nbody']

import sys
mach = sys.platform
if mach == 'linux':
    sys.path.append('/n/home01/abeane/mwib_analysis')
    base = '/n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo'
elif mach == 'darwin':
    sys.path.append('/Users/abeane/scratch/mwib_analysis')
    base = '/Users/abeane/scratch/mwib_runs/arepo'

print(base)

from mwib_analysis import mwib_io, mwib_agama, disk
import numpy as np
import h5py as h5
from tqdm import tqdm
import pickle
import astropy.units as u

sims_list = ['/galakos/lvl5', '/galakos/lvl4', '/galakos/lvl3-hernquist', '/galakos-fg0.2/lvl5']
path_list = [base+s for s in sims_list]
name_list = ['lvl5', 'lvl4', 'lvl3', 'lvl5-fg0.2']

dsnap2 = 100

snaps = ['100', '300', '500', '700', '900', '1100']
mlist = [2, 3, 4, 6, 8]

idx_list_list = [np.arange(int(sp)-dsnap2, int(sp)+dsnap2) for sp in snaps]

Rmin = 0.01
Rmax = 20.
nbins = 100

xlim = [0, 20]

for path, name in zip(path_list, name_list):
    indices, files = mwib_io.get_files_indices(path + '/output/*.hdf5')

    for snp, idx_list in zip(tqdm(snaps), idx_list_list):
        for m in mlist:
            to_dump = {}

            key = int(snp)

            f = files[key]

            time = mwib_io.get_time(f)
            time = "{:0.0f}".format(time)

            # compute and plot the spectrogram
            out, extent, time, Rbins = \
                disk.compute_spectrogram(path+'/output', idx_list, 'PartType2', 
                    m, Rmin, Rmax, nbins, Rlogspace=False, return_Rbins=True)

            to_dump['out'] = out
            to_dump['extent'] = extent
            to_dump['time'] = time

            # now get information needed for resonances
            try:
                pot_list = mwib_agama.read_potential_of_all_types(path+'/analysis/agama_pot/snapshot_'+"{:03d}".format(key))
            except:
                print('constructing potential for snap: ', f)
                snap = mwib_io.read_snap(f)
                pot_list = mwib_agama.construct_potential_of_all_types(snap)

            ptype_all = ('PartType0', 'PartType1', 'PartType2', 'PartType3', 'PartType4')
            vcirc = disk.get_vcirc_squared(None, ptype_all, pot_list, Rbins)
            vcirc = np.sqrt(vcirc)
            kappa = disk.get_radial_epicyclic_frequency(ptype_all, pot_list, Rbins)
            kappa = kappa / u.Myr
            kappa = kappa.to_value(u.km/u.s/u.kpc)

            to_dump['Rbins'] = Rbins
            to_dump['vcirc'] = vcirc
            to_dump['kappa'] = kappa

            to_dump['name'] = name
            to_dump['snap'] = snp
            to_dump['m'] = m

            fout = 'data/data_'+name+'_snap'+snp+'_m'+str(m)+'.p'
            print('writing to: ', fout)
            pickle.dump(to_dump, open(fout, 'wb'))
