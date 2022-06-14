import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import os

import h5py as h5

from joblib import Parallel, delayed

def read_snap(idx, name, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],
              basepath = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'):
    
    fname = basepath + name + '/output'
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

def _run_chunk(name, idx):
    sn = read_snap(idx, name.replace('-lvl', '/lvl'), parttype=None, fields=['Coordinates', 'Potential'])

    tot_pot_min = 1E99
    tot_pot_min_center = -1

    for i in range(len(sn.MassTable)):
        if sn.NumPart_Total[i] > 0:
            part = getattr(sn, 'part'+str(i))
            pot_min, key = np.min(part.pot.value), np.argmin(part.pot.value)

            if pot_min < tot_pot_min:
                tot_pot_min = pot_min
                tot_pot_min_center = part.pos.value[key]
    
    return tot_pot_min_center
    

def run(path, name, nsnap, nproc, phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data/'):
    prefix = 'data/pot_center_' + name + '.hdf5'

    # do standard fourier and bar angle stuff
    centers = Parallel(n_jobs=nproc) (delayed(_run_chunk)(name, i) for i in tqdm(range(nsnap)))

    out = h5.File(prefix, mode='w')
    out.create_dataset('PotentialCenter', data=centers)
    out.close()
    
    return

        
if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl4'), (Nbody, 'lvl3'),
                 (phgvS2Rc35, 'lvl4'), (phgvS2Rc35, 'lvl3'),
                 (phgvS2Rc35, 'lvl3-rstHalo')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]

    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    if len(sys.argv) == 3:
        i = int(sys.argv[2])
        path = path_list[i]
        name = name_list[i]
        nsnap = nsnap_list[i]

        out = run(path, name, nsnap, nproc)
    else:
        for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
            out = run(path, name, nsnap, nproc)
