import numpy as np
import arepo
import sys
from tqdm import tqdm
import h5py as h5
import glob
import os
import re
import time

from joblib import Parallel, delayed

def read_snap(path, idx, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs']):
    
    fname = path + '/output'
    
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

def _run_thread(path, name, idx_list, snap_id, id_chunks, data_dir='data_tmp/'):
    
    h5out_list = []

    Nsnap = len(idx_list)
    
    # Create a temporary directory which will store each chunk of ids as a separate file.
    prefix = data_dir + name + '/tmp' + str(snap_id)
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    
    # Loop through each id chunk and create the file with temporary output.
    for i,id_chunk_list in enumerate(id_chunks):
        Nids = len(id_chunk_list)

        fout = prefix + '/tmp' + str(i) + '.hdf5'
        h5out = h5.File(fout, mode='w')

        pos = np.zeros((Nids, Nsnap, 3))
        time = np.zeros(Nsnap)
        
        h5out.create_dataset("ParticleIDs", data=id_chunk_list)
        h5out.create_dataset("Time", data=time)
        h5out.create_dataset("Coordinates", data=pos)
        h5out.create_dataset("Velocities", data=pos)
        h5out.create_dataset("Acceleration", data=pos)

        h5out_list.append(h5out)
    
    # Now loop through each index, read the snapshot, then loop through each
    # id chunk and write to the relevant file.
    for i,idx in enumerate(idx_list):
        sn = read_snap(path, idx, parttype=[2], 
                       fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs', 'Acceleration'])
        # Sort by ID
        key = np.argsort(sn.part2.id)
        pos_ = sn.part2.pos.value[key]
        vel_ = sn.part2.vel.value[key]
        acc_ = sn.part2.acce[key]

        for j,id_chunk_list in enumerate(id_chunks):
            in_key = np.isin(np.sort(sn.part2.id), id_chunk_list)

            h5out_list[j]['Coordinates'][:,i,:] = pos_[in_key]
            h5out_list[j]['Velocities'][:,i,:] = vel_[in_key]
            h5out_list[j]['Acceleration'][:,i,:] = acc_[in_key]
            h5out_list[j]['Time'][i] = sn.Time.value
    
    # Close h5 files.
    for i,_ in enumerate(id_chunks):
        h5out_list[i].close()
    
    return None

def _concat_h5_files(name, chunk_id, id_chunk_list, indices_chunks, nsnap, data_dir='data_tmp/'):
    # First create hdf5 output file
    fout = data_dir + name + '/phase_space_' + name + '.' + str(chunk_id) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    # Temporary arrays for storing output
    Nids = len(id_chunk_list)

    pos = np.zeros((Nids, nsnap, 3))
    vel = np.zeros((Nids, nsnap, 3))
    acc = np.zeros((Nids, nsnap, 3))
    time = np.zeros(nsnap)

    # Prefix for temporary data files.
    prefix = data_dir + name + '/tmp'

    for j,idx_list in enumerate(indices_chunks):
        fin = prefix + str(j) + '/tmp' + str(chunk_id) + '.hdf5'
        h5in = h5.File(fin, mode='r')

        pos[:,idx_list,:] = np.array(h5in['Coordinates'])
        vel[:,idx_list,:] = np.array(h5in['Velocities'])
        acc[:,idx_list,:] = np.array(h5in['Acceleration'])
        time[idx_list] = np.array(h5in['Time'])

        h5in.close()

    h5out.create_dataset("ParticleIDs", data=id_chunk_list)
    h5out.create_dataset("Coordinates", data=pos)
    h5out.create_dataset("Velocities", data=vel)
    h5out.create_dataset("Acceleration", data=acc)
    h5out.create_dataset("Time", data=time)

    h5out.close()

def get_id_indices_chunks(nsnap, path, nchunk, nproc):
    indices = np.arange(nsnap)

    sn = read_snap(path, indices[-1], parttype=[2])
    ids = sn.part2.id
    ids = np.sort(ids)

    id_chunks = np.array_split(ids, nchunk)

    indices_chunks = np.array_split(indices, 4*nproc)

    return id_chunks, indices_chunks

def run(path, name, nsnap, nproc, nchunk, data_dir='data_tmp/'):
    
    # Split up particle ids and snapshot indices into chunks to be processed individually
    id_chunks, indices_chunks = get_id_indices_chunks(nsnap, path, nchunk, nproc)

    # If output directory does not exist, make it
    if not os.path.isdir(data_dir+name):
        os.mkdir(data_dir+name)

    # Runs through each chunk of indices and reads the snapshot of each index. Each chunk of ids is written to a different temporary file.
    t0 = time.time()
    _ = Parallel(n_jobs=nproc) (delayed(_run_thread)(path, name, indices_chunks[i], i, id_chunks) for i in tqdm(range(len(indices_chunks))))
    t1 = time.time()
    print('First loop took', t1-t0, 's')

    # Runs through each chunk of ids and reads the temporary files from the previous step, then writes a single file spanning all indices
    # for each chunk of ids.
    t0 = time.time()
    _ = Parallel(n_jobs=nproc) (delayed(_concat_h5_files)(name, i, id_chunks[i], indices_chunks, nsnap) for i in tqdm(range(len(id_chunks))))
    t1 = time.time()
    print('Second loop took', t1-t0, 's')

    return None    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl4', 64), (Nbody, 'lvl3', 2*64),
                 (phgvS2Rc35, 'lvl4', 64), (phgvS2Rc35, 'lvl3', 2*64)]

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
