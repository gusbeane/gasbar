import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import os
import pickle
import h5py as h5
from numba import njit

from joblib import Parallel, delayed

nres = 512
rng = [[-10, 10], [-10, 10]]

in_bar_path = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/in_bar/data/'
rcut = 2.67074

def get_center(sn):
    center = sn.part1.pos.value[np.argmin(sn.part1.pot.value)]
    return center

def read_in_bar(name, nchunk=256):
    fname_base = in_bar_path+'in_bar_'+name+'/in_bar_'+name+'.'
    
    in_bar = []
    
    for i in tqdm(range(nchunk)):
        fname = fname_base + str(i) + '.hdf5'
        
        t = h5.File(fname, mode='r')
        in_bar.append(t['PartType1']['in_bar'][:])
        t.close()
    
    in_bar = np.concatenate(in_bar, axis=1)
    
    return in_bar

def _runner(path, in_bar_i, snap, ptypes=[1]):
    sn = arepo.Snapshot(path + '/output/', snap, 
                        parttype=ptypes, 
                        fields=['Coordinates', 'Velocities', 'Potential', 'ParticleIDs'],
                        combineFiles=True)
    
    center = get_center(sn)
    
    part1_pos = sn.part1.pos.value - center
    part1_pos = part1_pos[np.argsort(sn.part1.id)]
    
    r = np.linalg.norm(part1_pos, axis=1)
    bool_rcut = r < rcut
    bool_2rcut = r < 2*rcut

    N_incut = len(np.where(np.logical_and(in_bar_i, bool_rcut))[0])
    N_2incut = len(np.where(np.logical_and(in_bar_i, bool_2rcut))[0])
    Ntot = len(part1_pos)
    ftrap = N_incut/Ntot
    ftrap2 = N_2incut/Ntot
    
    Time = sn.Time.value
    
    
    # Package it all together
    output = (Time, ftrap, ftrap2)
    
    return output

def run(path, name, nproc):
    
    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    
    in_bar = read_in_bar(name)
    
    out = Parallel(n_jobs=nproc) (delayed(_runner)(path, in_bar[i], i) for i in tqdm(range(nsnap)))

    Time = np.array([out[i][0] for i in range(len(out))])
    ftrap = np.array([out[i][1] for i in range(len(out))])
    ftrap2 = np.array([out[i][2] for i in range(len(out))])
    
    out = {'Time'  : Time,
           'ftrap' : ftrap,
           'ftrap2': ftrap2}
    
    np.save('halo_trapped_'+name+'.npy', out)

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../'

    Nbody = 'Nbody'
    SMUGGLE = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl3', 0), # 0
                 (SMUGGLE, 'lvl3', 0), # 8
                ]


    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + 'runs/' + p[0] + '/' + p[1] for p in pair_list]
    Nang_list  = [p[2] for p in pair_list]
    # ic_list   = [basepath + 'ics/' + p[0] + '/' + p[1] for p in pair_list]
    
  
    i = int(sys.argv[2])
    path = path_list[i]
    name = name_list[i]
    Nang = Nang_list[i]

    if Nang > 0:
        out = run_ang(path, name, nproc, Nang)
    else:
        out = run(path, name, nproc)
