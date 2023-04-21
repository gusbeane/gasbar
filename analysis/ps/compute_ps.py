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

def pattern_speed(sn, center, Rcut=3):
    pos = sn.part2.pos.value - center
    vel = sn.part2.vel.value
    
    R = np.linalg.norm(pos[:,:2], axis=1)
    # key = np.logical_and(R < Rcut, R > 0.005)
    # key = R < Rcut
    
    # print(len(np.where(key)[0]))
    
    key = np.logical_and(R > 0.01, R < Rcut) # R>0 is in case disk particle is center
    # key = np.logical_and(key, R > 0.01)
    pos = pos[key]
    vel = vel[key]
    R = R[key]
    
    phi = np.arctan2(pos[:,1], pos[:,0])
    
    cos = pos[:,0]/R
    sin = pos[:,1]/R
    
    vphi = vel[:,1] * cos - vel[:,0] * sin
    
    Re = np.sum(np.cos(2*phi))
    Im = np.sum(np.sin(2*phi))
    
    Redot = np.sum(- 2*vphi/R * np.sin(2*phi))
    Imdot = np.sum(2*vphi/R * np.cos(2*phi))
    
    ps = Imdot*Re - Redot*Im
    ps /= 2 * (Im**2 + Re**2)
    
    return ps

def get_center(name, sn):
    
    pot_min_1 = np.min(sn.part1.pot.value)
    pot_min_2 = np.min(sn.part2.pot.value)
    pot_min_3 = np.min(sn.part3.pot.value)
    
    center_1 = sn.part1.pos.value[np.argmin(sn.part1.pot.value)]
    center_2 = sn.part2.pos.value[np.argmin(sn.part2.pot.value)]
    center_3 = sn.part3.pos.value[np.argmin(sn.part3.pot.value)]
    
    center = center_1
    pot_min = pot_min_1
    
    if pot_min_2 < pot_min:
        center = center_2
        pot_min = pot_min_2
        
    if pot_min_3 < pot_min:
        center = center_3
        pot_min = pot_min_3
    
    return center

def _runner(path, name, snap, ptypes=[1, 2, 3]):
    sn = arepo.Snapshot(path + '/output/', snap, 
                        parttype=ptypes, 
                        fields=['Coordinates', 'Velocities', 'Potential'],
                        combineFiles=True)
    
    center = get_center(name, sn)
    
    ps = pattern_speed(sn, center)
    Time = sn.Time.value

    # Package it all together
    output = (ps, Time)
    
    return output

def run(path, name, nproc):
    
    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    
    out = Parallel(n_jobs=nproc) (delayed(_runner)(path, name, i) for i in tqdm(range(nsnap)))

    PatternSpeed = np.array([out[i][0] for i in range(len(out))])
    Time         = np.array([out[i][1] for i in range(len(out))])

    out = {'PatternSpeed' : PatternSpeed,
           'Time'          : Time}
    
    np.save('ps_'+name+'.npy', out)

def run_ang(path, name, nproc, Nang):
    ang_list = np.arange(0, 1, 1/Nang)
    
    for ang in ang_list:
        this_name = name + '-ang' + str(ang)
        this_path = path + '-ang' + str(ang)
        
        run(this_path, this_name, nproc)

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
