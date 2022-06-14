import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import h5py as h5
import glob
import os
from numba import njit
import re
from sklearn.cluster import KMeans

from joblib import Parallel, delayed

def read_snap(path, idx, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs']):
    
    fname = path + '/output'
    
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

@njit
def process_id(ID, bar_metr, t, tlist, nsnap):
    in_bar = np.full(nsnap, 0, dtype=np.bool_)
    Naps = t.shape[0]
    for j,idx in enumerate(range(nsnap)):
        Tanalyze = tlist[idx]
        key = np.argmin(np.abs(t - Tanalyze))

        if key==0 or key==Naps-1:
            continue
        
        metr = bar_metr[key]
        
        if metr[3] == 0.0:
            continue
        
        c0 = metr[1] < np.pi/8.0
        c1 = metr[2]/metr[3] < 0.22
        if c0 and c1:
            in_bar[j] = 1
    
    return in_bar

def _in_bar_one_chunk(prefix_in, prefix_out, name, chunk_idx):
    fin = prefix_in+'/bar_orbit_'+name+'.' + str(chunk_idx) + '.hdf5'
    h5in = h5.File(fin, mode='r')

    idx_list = np.array(h5in['idx_list'])
    nsnap = len(idx_list)
    
    id_list = np.array(h5in['id_list']).astype(np.int)
    tlist = np.array(h5in['tlist'])
    bar_angle = np.array(h5in['bar_angle'])
    
    in_bar = np.full((nsnap, len(id_list)), 0, dtype=np.bool_)
    
    for i,ID in enumerate(id_list):
        bar_metr = np.array(h5in['bar_metrics'][str(ID)])
        t = tlist[bar_metr[:,0].astype(np.int)]
        in_bar[:,i] = process_id(ID, bar_metr, t, tlist, nsnap)
    
    fout = prefix_out + '/in_bar_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    # write to output file
    h5out.create_dataset('tlist', data=tlist)
    h5out.create_dataset('id_list', data=id_list)
    h5out.create_dataset('idx_list', data=idx_list)
    h5out.create_dataset('bar_angle', data=bar_angle)

    h5out.create_dataset('in_bar', data=in_bar)

    h5in.close()
    h5out.close()
    
    return None

def run(name, nproc, basepath = '/n/home01/abeane/starbar/plots/bar_orbits/data/'):
    prefix_out = 'data/in_bar_' + name + '/'
    if not os.path.isdir(prefix_out):
        os.mkdir(prefix_out)

    prefix_in = basepath + 'bar_orbit_' + name
    nchunk = len(glob.glob(prefix_in+'/bar_orbit_'+name+'.*.hdf5'))
    
    _ = Parallel(n_jobs=nproc)(delayed(_in_bar_one_chunk)(prefix_in, prefix_out, name, i) for i in tqdm(range(nchunk)))
    
    return None        

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

    # nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    if len(sys.argv) == 3:
        i = int(sys.argv[2])
        path = path_list[i]
        name = name_list[i]
        # nsnap = nsnap_list[i]

        out = run(name, nproc)
    else:
        for path, name in zip(tqdm(path_list), name_list):
            out = run(name, nproc)
