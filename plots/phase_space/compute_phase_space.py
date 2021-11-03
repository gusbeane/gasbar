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

def construct_orbits(path, idx):
    sn = read_snap(path, idx, parttype=[2])
    
    id_list = sn.part2.id
    keys = np.argsort(id_list)
    
    pos = sn.part2.pos.value
    pos = pos[keys]
    
    vel = sn.part2.vel.value
    vel = vel[keys]
    
    w = np.hstack((pos, vel))
    t = sn.Time.value

    return t, w

def _run_thread(path, name, idx_list, snap_id, id_chunks):
    h5out_list = []
    prefix = 'data_tmp/' + name + '/tmp' + str(snap_id)
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    for i,id_chunk_list in enumerate(id_chunks):
        fout = prefix + '/tmp' + str(i) + '.hdf5'
        h5out = h5.File(fout, mode='w')
        h5out.create_dataset("ParticleIDs", data=id_chunk_list)

        pos = np.zeros((len(id_chunk_list), len(idx_list), 3))

        h5out.create_dataset("Coordinates", data=pos)
        h5out.create_dataset("Velocities", data=pos)

        time = np.zeros(len(idx_list))
        h5out.create_dataset("Time", data=time)

        h5out_list.append(h5out)
    
    for i,idx in enumerate(idx_list):
        sn = read_snap(path, idx, parttype=[2])
        key = np.argsort(sn.part2.id)
        pos_ = sn.part2.pos.value[key]
        vel_ = sn.part2.vel.value[key]

        for j,id_chunk_list in enumerate(id_chunks):
            in_key = np.isin(np.sort(sn.part2.id), id_chunk_list)
            h5out_list[j]['Coordinates'][:,i,:] = pos_[in_key]
            h5out_list[j]['Velocities'][:,i,:] = vel_[in_key]
            h5out_list[j]['Time'][i] = sn.Time.value
        
    for i,_ in enumerate(id_chunks):
        h5out_list[i].close()
    
    return None

def _concat_h5_files(name, chunk_id, id_chunk_list, indices_chunks, nsnap):
    # first create dummy hdf5 files
    prefix = 'data_tmp/' + name + '/tmp'

    fout = 'data_tmp/' + name + '/phase_space_' + name + '.' + str(chunk_id) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    h5out.create_dataset("ParticleIDs", data=id_chunk_list)

    pos = np.zeros((len(id_chunk_list), nsnap, 3))
    vel = np.zeros((len(id_chunk_list), nsnap, 3))
    time = np.zeros(nsnap)

    # h5out.create_dataset("Coordinates", data=pos)
    # h5out.create_dataset("Velocities", data=pos)

    for j,idx_list in enumerate(indices_chunks):
        fin = prefix + str(j) + '/tmp' + str(chunk_id) + '.hdf5'
        h5in = h5.File(fin, mode='r')

        print(idx_list)
        print(np.shape(h5in['Coordinates']))

        pos[:,idx_list,:] = np.array(h5in['Coordinates'])
        vel[:,idx_list,:] = np.array(h5in['Velocities'])
        time[idx_list] = np.array(h5in['Time'])

        h5in.close()

    h5out.create_dataset("Coordinates", data=pos)
    h5out.create_dataset("Velocities", data=vel)
    h5out.create_dataset("Time", data=time)

    h5out.close()

def run(path, name, nsnap, nproc, nchunk):
    indices = np.arange(nsnap)

    sn = read_snap(path, indices[-1], parttype=[2])
    ids = sn.part2.id
    ids = np.sort(ids)

    id_chunks = np.array_split(ids, nchunk)

    indices_chunks = np.array_split(indices, 4*nproc)

    if not os.path.isdir('data_tmp/'+name):
        os.mkdir('data_tmp/'+name)

    _ = Parallel(n_jobs=nproc) (delayed(_run_thread)(path, name, indices_chunks[i], i, id_chunks) for i in tqdm(range(len(indices_chunks))))

    _ = Parallel(n_jobs=nproc) (delayed(_concat_h5_files)(name, i, id_chunks[i], indices_chunks, nsnap) for i in tqdm(range(len(id_chunks))))

    # concat_h5_files(name, id_chunks, indices_chunks, nsnap)

    return None


# def run(path, name, nsnap, nproc, nchunk):

#     indices = np.arange(nsnap)
    
#     sn = read_snap(path, indices[-1], parttype=[2])
#     ids = sn.part2.id
#     ids = np.sort(ids)

#     # open a file and load dummy data for each chunk of stars
#     id_chunks = np.array_split(ids, nchunk)
#     h5out_list = []
#     for i,id_chunk_list in tqdm(enumerate(id_chunks)):
#         fout = 'data/phase_space_' + name + '.' + str(i) + '.hdf5'
#         h5out = h5.File(fout, mode='w')
#         h5out.create_dataset("ParticleIDs", data=id_chunk_list)
        
#         pos = np.zeros((len(id_chunk_list), nsnap, 3))
#         vel = np.zeros((len(id_chunk_list), nsnap, 3))

#         h5out.create_dataset("Coordinates", data=pos)
#         h5out.create_dataset("Velocities", data=vel)

#         h5out_list.append(h5out)
    
#     # now load each snapshot and modify the h5 files

#     for idx in tqdm(indices):
#         sn = read_snap(path, idx, parttype=[2])
#         key = np.argsort(sn.part2.id)
#         pos_ = sn.part2.pos.value[key]
#         vel_ = sn.part2.vel.value[key]

#         for i,id_chunk_list in enumerate(id_chunks):
#             in_key = np.isin(np.sort(sn.part2.id), id_chunk_list)
#             h5out_list[i]['Coordinates'][:,idx,:] = pos_[in_key]
#             h5out_list[i]['Velocities'][:,idx,:] = vel_[in_key]
        
#     for i,_ in enumerate(id_chunks):
#         h5out_list[i].close()
    
#     return None
    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl4', 64), (Nbody, 'lvl3', 64*8)]
                #  (phgvS2Rc35, 'lvl4'), (phgvS2Rc35, 'lvl3')]

    name_list   = [           p[0] + '-' + p[1] for p in pair_list]
    path_list   = [basepath + p[0] + '/' + p[1] for p in pair_list]
    nchunk_list = [ p[2] for p in pair_list ]

    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    if len(sys.argv) == 3:
        i = int(sys.argv[2])
        path = path_list[i]
        name = name_list[i]
        nsnap = nsnap_list[i]
        nchunk = nchunk_list[i]

        out = run(path, name, nsnap, nproc, nchunk)
    else:
        for path, name, nsnap, nchunk in zip(tqdm(path_list), name_list, nsnap_list, nchunk_list):
            out = run(path, name, nsnap, nproc, nchunk)
