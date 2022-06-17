import numpy as np
import arepo
import sys
from tqdm import tqdm
import h5py as h5
import glob
import os
import re
import time
from numba import njit

import cProfile

from joblib import Parallel, delayed

def _concat_h5_files(name, lvl, chunk_idx, data_dir='data/'):
    fields = ['Coordinates', 'Velocities', 'Acceleration']

    # First create hdf5 output file
    prefix = data_dir + name + '-' + lvl + '/'
    fout = prefix + 'phase_space_' + name + '-' + lvl + '.' + str(chunk_idx) + '.hdf5'

    nsnap = len(glob.glob(prefix+'/tmp*/tmp'+str(chunk_idx)+'.hdf5'))

    first_fname = prefix + '/tmp0/tmp'+str(chunk_idx)+'.hdf5'
    first_h5 = h5.File(first_fname, mode='r')
    out = {}

    ptypes = list(first_h5.keys())
    for pt in ptypes:
        out[pt] = {}
        Nids = len(first_h5[pt]['ParticleIDs'])
        for fld in fields:
            # out[pt][fld] = np.array([], dtype=np.float64).reshape(Nids, 0, 3)
            out[pt][fld] = []

        out[pt]['ParticleIDs'] = first_h5[pt]['ParticleIDs'][:]
        out['Time'] = np.zeros(nsnap)
    first_h5.close()

    for i in range(nsnap):
        fname = prefix + '/tmp'+str(i)+'/tmp'+str(chunk_idx)+'.hdf5'
        h5in = h5.File(fname, mode='r')

        for pt in ptypes:
            for fld in fields:
                out[pt][fld].append(np.expand_dims(h5in[pt][fld][:], 1))
        
        out['Time'][i] = h5in.attrs['Time']

        h5in.close()
    
    for pt in ptypes:
        for fld in fields:
            out[pt][fld] = np.concatenate(out[pt][fld], axis=1)

    # create output file
    h5out = h5.File(fout, mode='w')
    for pt in ptypes:
        h5out.create_dataset(pt + '/ParticleIDs', data=out[pt]['ParticleIDs'])
        for fld in fields:
            h5out.create_dataset(pt + '/' + fld, data=out[pt][fld])
    
    h5out.create_dataset('Time', data=out['Time'])

    h5out.close()

def run(name, lvl, nchunk, nproc=1, data_dir='data/'):
    
    print('running ', name, lvl)

    _ = Parallel(n_jobs=nproc) (delayed(_concat_h5_files)(name, lvl, i) for i in tqdm(range(nchunk)))

    # for i in tqdm(range(nchunk)):
        # _concat_h5_files(name, lvl, i)

    # _concat_h5_files(name, lvl, 0)

    return None

if __name__ == '__main__':
    name = sys.argv[1]
    lvl = sys.argv[2]
    nproc = int(sys.argv[3])

    nchunk = 256

    run(name, lvl, nchunk, nproc)