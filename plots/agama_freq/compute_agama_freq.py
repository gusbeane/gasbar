import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import os
import h5py as h5
import shutil

from joblib import Parallel, delayed

import agama
agama.setUnits(mass=1E10, length=1, velocity=1)

NPART = 6

def read_snap(idx, name, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],
              basepath = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'):
    
    fname = basepath + name + '/output'
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

def _run_chunk(name, idx, prefix, pot_prefix, center):
    sn = read_snap(idx, name.replace('-lvl', '/lvl'), parttype=None)

    agama.setUnits(mass=1E10, length=1, velocity=1)

    try:
        pot_fname = pot_prefix + 'pot_' + name + '.' + str(idx) + '.txt'
        pot = agama.Potential(pot_fname)
        af = agama.ActionFinder(pot)
    except:
        # if fails, warn and use previous
        print('WARNING: potential at idx ', idx, 'failed to create action finer, using idx ', idx-1)
        pot_fname = pot_prefix + 'pot_' + name + '.' + str(idx-1) + '.txt'
        pot = agama.Potential(pot_fname)
        af = agama.ActionFinder(pot)

    fout = prefix + 'freq_'+ name + '.' + str(idx) + '.hdf5'
    tmpout = '/scratch/' + 'freq_' + name + '.' + str(idx) + '.hdf5'
    h5out = h5.File(tmpout, mode='w')

    for i in range(NPART):
        if sn.NumPart_Total[i] > 0:
            part = getattr(sn, 'part'+str(i))
            
            pos = part.pos.value - center
            vel = part.vel.value
            w = np.hstack((pos, vel))

            # sort by id
            w = w[np.argsort(part.id)]

            act, ang, freq = af(w, angles=True)

            h5out.create_dataset('PartType'+str(i)+'/Actions', data=act)
            h5out.create_dataset('PartType'+str(i)+'/Angles', data=ang)
            h5out.create_dataset('PartType'+str(i)+'/Frequencies', data=freq)
    
    h5out.create_dataset('Header/MassTable', data=sn.MassTable)
    h5out.create_dataset('Header/NumPart_Total', data=sn.NumPart_Total)
    h5out.create_dataset('Header/Time', data=sn.Time)
    
    h5out.close()

    shutil.copy(tmpout, fout)
    os.remove(tmpout)

def preprocess_center(name):
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
        firstkey=150
        indices = np.arange(nsnap)
        # indices_analyze = np.arange(500, 1100, 20)
    else:
        center = np.array([200, 200, 200])
        firstkey=0
        indices = np.arange(nsnap)

    return center, firstkey, indices

def run(path, name, nsnap, nproc, phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data/'):
    prefix = 'data/freq_' + name +'/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    
    pot_prefix = '../agama_pot/data/pot_' + name + '/'

    # get some preliminary variables
    center, firstkey, indices = preprocess_center(name)
    
    # do standard fourier and bar angle stuff
    _ = Parallel(n_jobs=nproc) (delayed(_run_chunk)(name, i, prefix, pot_prefix, center) for i in tqdm(indices))
        
    # for i in tqdm(indices):
    #    print(i)
    #    _run_chunk(name, i, prefix, pot_prefix, center)

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
