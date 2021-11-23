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

from joblib import Parallel, delayed

def read_snap(path, idx, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs']):
    
    fname = path + '/output'
    
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

@njit
def sort_by_id(chunk_ids, tot_ids, pos, vel, acc):
    # This goes through the chunk ids and matches up with total ids
    # but this assumes chunk ids and total ids are already sorted, which greatly
    # speeds things up.

    # also properly handles missing ids (e.g. in the case of stars)

    Nchunk = len(chunk_ids)
    pos_chunk = np.zeros((Nchunk, 3))
    vel_chunk = np.zeros((Nchunk, 3))
    acc_chunk = np.zeros((Nchunk, 3))
    
    itot = 0
    
    for ichunk in range(Nchunk):
        chk_id = chunk_ids[ichunk]
        
        while chk_id > tot_ids[itot]:
            itot += 1
        
        if chk_id == tot_ids[itot]:
            for j in range(3):
                pos_chunk[ichunk][j] = pos[itot][j]
                vel_chunk[ichunk][j] = vel[itot][j]
                acc_chunk[ichunk][j] = acc[itot][j]
        
        else:
            for j in range(3):
                pos_chunk[ichunk][j] = np.nan
                vel_chunk[ichunk][j] = np.nan
                acc_chunk[ichunk][j] = np.nan
        
        itot += 1
    
    return pos_chunk, vel_chunk, acc_chunk


def _run_thread(path, name, idx_list, snap_id, id_chunks_disk, id_chunks_bulge, data_dir='data_tmp/'):
    
    h5out_list = []

    Nsnap = len(idx_list)
    
    # Create a temporary directory which will store each chunk of ids as a separate file.
    prefix = data_dir + name + '/tmp' + str(snap_id)
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    
    # Loop through each id chunk and create the file with temporary output.
    for i,(id_chunk_disk_list, id_chunk_bulge_list) in enumerate(zip(id_chunks_disk, id_chunks_bulge)):
        Nids_disk = len(id_chunk_disk_list)
        Nids_bulge = len(id_chunk_bulge_list)

        fout = prefix + '/tmp' + str(i) + '.hdf5'
        h5out = h5.File(fout, mode='w')

        pos_disk = np.zeros((Nids_disk, Nsnap, 3))
        pos_bulge = np.zeros((Nids_bulge, Nsnap, 3))
        time = np.zeros(Nsnap)
        
        h5out.create_dataset("Time", data=time)

        h5out.create_dataset("PartType2/ParticleIDs", data=id_chunk_disk_list)
        h5out.create_dataset("PartType2/Coordinates", data=pos_disk)
        h5out.create_dataset("PartType2/Velocities", data=pos_disk)
        h5out.create_dataset("PartType2/Acceleration", data=pos_disk)

        h5out.create_dataset("PartType3/ParticleIDs", data=id_chunk_bulge_list)
        h5out.create_dataset("PartType3/Coordinates", data=pos_bulge)
        h5out.create_dataset("PartType3/Velocities", data=pos_bulge)
        h5out.create_dataset("PartType3/Acceleration", data=pos_bulge)

        h5out_list.append(h5out)
    
    # Now loop through each index, read the snapshot, then loop through each
    # id chunk and write to the relevant file.
    for i,idx in enumerate(idx_list):
        sn = read_snap(path, idx, parttype=[2, 3], 
                       fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs', 'Acceleration'])
        # Sort by ID
        key_disk = np.argsort(sn.part2.id)
        pos_disk = sn.part2.pos.value[key_disk]
        vel_disk = sn.part2.vel.value[key_disk]
        acc_disk = sn.part2.acce[key_disk]

        key_bulge = np.argsort(sn.part3.id)
        pos_bulge = sn.part3.pos.value[key_bulge]
        vel_bulge = sn.part3.vel.value[key_bulge]
        acc_bulge = sn.part3.acce[key_bulge]

        disk_ids_sorted = sn.part2.id[key_disk]
        bulge_ids_sorted = sn.part3.id[key_bulge]

        for j,(id_chunk_disk_list, id_chunk_bulge_list) in enumerate(zip(id_chunks_disk, id_chunks_bulge)):
            pos_chunk_disk, vel_chunk_disk, acc_chunk_disk = sort_by_id(id_chunk_disk_list, disk_ids_sorted, pos_disk, vel_disk, acc_disk)
            pos_chunk_bulge, vel_chunk_bulge, acc_chunk_bulge = sort_by_id(id_chunk_bulge_list, bulge_ids_sorted, pos_bulge, vel_bulge, acc_bulge)

            h5out_list[j]['Time'][i] = sn.Time.value

            h5out_list[j]['PartType2/Coordinates'][:,i,:] = pos_chunk_disk
            h5out_list[j]['PartType2/Velocities'][:,i,:] = vel_chunk_disk
            h5out_list[j]['PartType2/Acceleration'][:,i,:] = acc_chunk_disk

            h5out_list[j]['PartType3/Coordinates'][:,i,:] = pos_chunk_bulge
            h5out_list[j]['PartType3/Velocities'][:,i,:] = vel_chunk_bulge
            h5out_list[j]['PartType3/Acceleration'][:,i,:] = acc_chunk_bulge

    
    # Close h5 files.
    for i,_ in enumerate(id_chunks_disk):
        h5out_list[i].close()
    
    return None

def _concat_h5_files(name, chunk_id, id_chunk_disk_list, id_chunk_bulge_list, indices_chunks, nsnap, data_dir='data_tmp/'):
    # First create hdf5 output file
    fout = data_dir + name + '/phase_space_' + name + '.' + str(chunk_id) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    # Temporary arrays for storing output
    Nids_disk = len(id_chunk_disk_list)
    Nids_bulge = len(id_chunk_bulge_list)

    time = np.zeros(nsnap)

    pos_disk = np.zeros((Nids_disk, nsnap, 3))
    vel_disk = np.zeros((Nids_disk, nsnap, 3))
    acc_disk = np.zeros((Nids_disk, nsnap, 3))

    pos_bulge = np.zeros((Nids_bulge, nsnap, 3))
    vel_bulge = np.zeros((Nids_bulge, nsnap, 3))
    acc_bulge = np.zeros((Nids_bulge, nsnap, 3))


    # Prefix for temporary data files.
    prefix = data_dir + name + '/tmp'

    for j,idx_list in enumerate(indices_chunks):
        fin = prefix + str(j) + '/tmp' + str(chunk_id) + '.hdf5'
        h5in = h5.File(fin, mode='r')

        time[idx_list] = np.array(h5in['Time'])

        pos_disk[:,idx_list,:] = np.array(h5in['PartType2/Coordinates'])
        vel_disk[:,idx_list,:] = np.array(h5in['PartType2/Velocities'])
        acc_disk[:,idx_list,:] = np.array(h5in['PartType2/Acceleration'])

        pos_bulge[:,idx_list,:] = np.array(h5in['PartType3/Coordinates'])
        vel_bulge[:,idx_list,:] = np.array(h5in['PartType3/Velocities'])
        acc_bulge[:,idx_list,:] = np.array(h5in['PartType3/Acceleration'])

        h5in.close()

    h5out.create_dataset("Time", data=time)

    h5out.create_dataset("PartType2/ParticleIDs", data=id_chunk_disk_list)
    h5out.create_dataset("PartType2/Coordinates", data=pos_disk)
    h5out.create_dataset("PartType2/Velocities", data=vel_disk)
    h5out.create_dataset("PartType2/Acceleration", data=acc_disk)

    h5out.create_dataset("PartType3/ParticleIDs", data=id_chunk_disk_list)
    h5out.create_dataset("PartType3/Coordinates", data=pos_disk)
    h5out.create_dataset("PartType3/Velocities", data=vel_disk)
    h5out.create_dataset("PartType3/Acceleration", data=acc_disk)

    h5out.close()

def get_id_indices_chunks(nsnap, path, nchunk, nproc):
    indices = np.arange(nsnap)

    sn = read_snap(path, indices[-1], parttype=[2, 3])
    ids_disk = sn.part2.id
    ids_disk = np.sort(ids_disk)

    ids_bulge = sn.part3.id
    ids_bulge = np.sort(ids_bulge)

    id_chunks_disk = np.array_split(ids_disk, nchunk)
    id_chunks_bulge = np.array_split(ids_bulge, nchunk)

    indices_chunks = np.array_split(indices, 4*nproc)

    return id_chunks_disk, id_chunks_bulge, indices_chunks

def run(path, name, nsnap, nproc, nchunk, data_dir='data_tmp/'):
    
    # Split up particle ids and snapshot indices into chunks to be processed individually
    id_chunks_disk, id_chunks_bulge, indices_chunks = get_id_indices_chunks(nsnap, path, nchunk, nproc)

    # If output directory does not exist, make it
    if not os.path.isdir(data_dir+name):
        os.mkdir(data_dir+name)

    # Runs through each chunk of indices and reads the snapshot of each index. Each chunk of ids is written to a different temporary file.
    t0 = time.time()
    _ = Parallel(n_jobs=nproc) (delayed(_run_thread)(path, name, indices_chunks[i], i, id_chunks_disk, id_chunks_bulge) for i in tqdm(range(len(indices_chunks))))
    t1 = time.time()
    print('First loop took', t1-t0, 's')

    # Runs through each chunk of ids and reads the temporary files from the previous step, then writes a single file spanning all indices
    # for each chunk of ids.
    t0 = time.time()
    _ = Parallel(n_jobs=nproc) (delayed(_concat_h5_files)(name, i, id_chunks_disk[i], id_chunks_bulge[i], indices_chunks, nsnap) for i in tqdm(range(len(id_chunks_disk))))
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
