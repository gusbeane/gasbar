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
import shutil
from distutils.dir_util import copy_tree

import cProfile

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
        
    return pos_chunk, vel_chunk, acc_chunk

def _run_thread(path, name_lvl, snap_idx, id_chunks_disk, id_chunks_bulge, id_chunks_star, data_dir, tmp_dir):

    print('starting thread: ', snap_idx)

    h5out_list = []

    # Create a temporary directory which will store each chunk of ids as a separate file.
    prefix = tmp_dir + str(np.random.rand()) + '/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)
    
    if id_chunks_star is None:
        has_stars = False
        id_chunks_star = [None for i in range(len(id_chunks_disk))]
    else:
        has_stars = True

    # Loop through each id chunk and create the file with temporary output.
    for i,(id_chunk_disk_list, id_chunk_bulge_list, id_chunk_star_list) in enumerate(zip(id_chunks_disk, id_chunks_bulge, id_chunks_star)):
        Nids_disk = len(id_chunk_disk_list)
        Nids_bulge = len(id_chunk_bulge_list)
        if has_stars:
            Nids_star = len(id_chunk_star_list)

        fout = prefix + '/tmp' + str(i) + '.hdf5'
        h5out = h5.File(fout, mode='w')

        pos_disk = np.zeros((Nids_disk, 3))
        pos_bulge = np.zeros((Nids_bulge, 3))
        if has_stars:
            pos_star = np.zeros((Nids_star, 3))
        
        h5out.create_dataset("PartType2/ParticleIDs", data=id_chunk_disk_list)
        h5out.create_dataset("PartType2/Coordinates", data=pos_disk)
        h5out.create_dataset("PartType2/Velocities", data=pos_disk)
        h5out.create_dataset("PartType2/Acceleration", data=pos_disk)

        h5out.create_dataset("PartType3/ParticleIDs", data=id_chunk_bulge_list)
        h5out.create_dataset("PartType3/Coordinates", data=pos_bulge)
        h5out.create_dataset("PartType3/Velocities", data=pos_bulge)
        h5out.create_dataset("PartType3/Acceleration", data=pos_bulge)

        if has_stars:
            h5out.create_dataset("PartType4/ParticleIDs", data=id_chunk_star_list)
            h5out.create_dataset("PartType4/Coordinates", data=pos_star)
            h5out.create_dataset("PartType4/Velocities", data=pos_star)
            h5out.create_dataset("PartType4/Acceleration", data=pos_star)

        h5out_list.append(h5out)
    
    print('ended first loop on thread', snap_idx)

    # Now loop through each index, read the snapshot, then loop through each
    # id chunk and write to the relevant file.
    sn = read_snap(path, snap_idx, parttype=[2, 3, 4], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs', 'Acceleration'])

    key_disk = np.argsort(sn.part2.id)
    pos_disk = sn.part2.pos.value[key_disk]
    vel_disk = sn.part2.vel.value[key_disk]
    acc_disk = sn.part2.acce[key_disk]
    disk_ids_sorted = sn.part2.id[key_disk]

    key_bulge = np.argsort(sn.part3.id)
    pos_bulge = sn.part3.pos.value[key_bulge]
    vel_bulge = sn.part3.vel.value[key_bulge]
    acc_bulge = sn.part3.acce[key_bulge]
    bulge_ids_sorted = sn.part3.id[key_bulge]

    if has_stars and sn.NumPart_Total[4] > 0:
        key_star = np.argsort(sn.part4.id)
        pos_star = sn.part3.pos.value[key_star]
        vel_star = sn.part3.vel.value[key_star]
        acc_star = sn.part3.acce[key_star]
        
        star_ids_sorted = sn.part4.id[key_star]

    for j,(id_chunk_disk_list, id_chunk_bulge_list, id_chunk_star_list) in enumerate(zip(id_chunks_disk, id_chunks_bulge, id_chunks_star)):
        h5out_list[j].attrs.create('Time', sn.Time.value)
            
        pos_chunk_disk, vel_chunk_disk, acc_chunk_disk = sort_by_id(id_chunk_disk_list, disk_ids_sorted, pos_disk, vel_disk, acc_disk)
        h5out_list[j]['PartType2/Coordinates'][:] = pos_chunk_disk
        h5out_list[j]['PartType2/Velocities'][:] = vel_chunk_disk
        h5out_list[j]['PartType2/Acceleration'][:] = acc_chunk_disk
            
        pos_chunk_bulge, vel_chunk_bulge, acc_chunk_bulge = sort_by_id(id_chunk_bulge_list, bulge_ids_sorted, pos_bulge, vel_bulge, acc_bulge)
        h5out_list[j]['PartType3/Coordinates'][:] = pos_chunk_bulge
        h5out_list[j]['PartType3/Velocities'][:] = vel_chunk_bulge
        h5out_list[j]['PartType3/Acceleration'][:] = acc_chunk_bulge

        if has_stars:
            if sn.NumPart_Total[4] > 0:
                pos_chunk_star, vel_chunk_star, acc_chunk_star = sort_by_id(id_chunk_star_list, star_ids_sorted, pos_star, vel_star, acc_star)
            else:
                pos_chunk_star = vel_chunk_star = acc_chunk_star = np.full((len(id_chunk_star_list), 3), np.nan)
            h5out_list[j]['PartType4/Coordinates'][:] = pos_chunk_star
            h5out_list[j]['PartType4/Velocities'][:] = vel_chunk_star
            h5out_list[j]['PartType4/Acceleration'][:] = acc_chunk_star

    print('ended second loop on thread', snap_idx)

    # Close h5 files.
    for i,_ in enumerate(id_chunks_disk):
        h5out_list[i].close()
    
    print('closed h5 files on thread', snap_idx)

    # Now copy the tmp directory to the data directory
    copy_tree(prefix, data_dir+name_lvl+'/tmp' + str(snap_idx))

    return prefix

def get_id_indices_chunks(path, nchunk):
    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    indices = np.arange(nsnap)

    sn = read_snap(path, indices[-1], parttype=[2, 3, 4])
    ids_disk = sn.part2.id
    ids_disk = np.sort(ids_disk)

    ids_bulge = sn.part3.id
    ids_bulge = np.sort(ids_bulge)

    if sn.NumPart_Total[4] > 0:
        ids_star = sn.part4.id
        ids_star = np.sort(ids_star)
        id_chunks_star = np.array_split(ids_star, nchunk)
    else:
        id_chunks_star = None

    id_chunks_disk = np.array_split(ids_disk, nchunk)
    id_chunks_bulge = np.array_split(ids_bulge, nchunk)

    return id_chunks_disk, id_chunks_bulge, id_chunks_star

def run(name, lvl, snap_idx, nchunk, basepath='../../runs/', data_dir='data/', tmp_dir='/tmp/'):

    print('running ', name, lvl, )

    name_lvl = name + '-' + lvl
    path = basepath + name + '/' + lvl

    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    if snap_idx >= nsnap:
        print('dont have snap_idx ', snap_idx, 'quitting...')
        sys.exit(0)

    # Split up particle ids and snapshot indices into chunks to be processed individually
    id_chunks_disk, id_chunks_bulge, id_chunks_star = get_id_indices_chunks(path, nchunk)

    # If output directory does not exist, make it
    if not os.path.isdir(data_dir+name_lvl):
        os.mkdir(data_dir+name_lvl)

    # Runs through each chunk of indices and reads the snapshot of each index. Each chunk of ids is written to a different temporary file.
    t0 = time.time()
    to_delete = _run_thread(path, name_lvl, snap_idx, id_chunks_disk, id_chunks_bulge, id_chunks_star, data_dir, tmp_dir)
    t1 = time.time()
    print('First loop took', t1-t0, 's')

    shutil.rmtree(to_delete)

    return None

if __name__ == '__main__':
    name = sys.argv[1]
    lvl = sys.argv[2]
    snap_idx = int(sys.argv[3])

    nchunk = 256

    run(name, lvl, snap_idx, nchunk)
