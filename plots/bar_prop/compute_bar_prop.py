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

def compute_bar_properties(name, mass, in_bar_out, center):
    nsnap = np.shape(in_bar_out[0])[0]
    idx_list = np.arange(0, nsnap, 50)

    phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data/'
    nchunk = len(glob.glob(phase_space_path+name+'/phase_space_'+name+'.*.hdf5'))

    bar_prop_list = np.zeros((len(idx_list), 5))

    ctr = 0
    for i in tqdm(range(nchunk)):
        in_bar = in_bar_out[i]
        Mbar = 0.0
        Lzbar = 0.0
    
        fin = phase_space_path + name + '/phase_space_' + name + '.' + str(i) + '.hdf5'
        h5in = h5.File(fin, mode='r')
    
        tlist = np.array(h5in['Time'])
        pos_tot = np.array(h5in['Coordinates']) - center
        vel_tot = np.array(h5in['Velocities'])
        ids = np.array(h5in['ParticleIDs'])
    
        if i==0:
            Rlist = {}
            for idx in idx_list:
                Rlist[idx] = np.array([])
    
        for j,idx in enumerate(idx_list):
            pos = pos_tot[:,idx,:]
            vel = vel_tot[:,idx,:]
        
            key = in_bar[idx][ctr:ctr+len(ids)]
        
        
            pos_inbar = pos[key]
            vel_inbar = vel[key]

            N_inbar = len(np.where(key)[0])
    
            Mbar = mass * len(pos_inbar)
            Lzbar = mass * np.sum(np.cross(pos_inbar, vel_inbar)[:,2])
        
            bar_prop_list[j,0] = tlist[idx]
            bar_prop_list[j,3] += Mbar
            bar_prop_list[j,4] += Lzbar
        
            if N_inbar == 0:
                R = np.array([])
            elif N_inbar == 1:
                R = np.array([np.sqrt(pos_inbar[0][0]**2 + pos_inbar[0][1]**2)]).reshape((1,))
            else:
                R = np.linalg.norm(pos_inbar[:,:2], axis=1)
            Rlist[idx] = np.concatenate((Rlist[idx], R))
    
        ctr += len(ids)
    
    
    for j,idx in enumerate(idx_list):
        if len(Rlist[idx]>100):
            Rbar = np.percentile(Rlist[idx], 99)
        else:
            Rbar = 0.0
        bar_prop_list[j,2] = Rbar
    
    return bar_prop_list

def get_mass_and_center_from_name(name):
    if 'lvl5' in name:
        mass = 6E-6 * 8.0
    elif 'lvl4' in name:
        mass = 6E-6
    elif 'lvl3' in name:
        mass = 6E-6 / 8.0
    elif 'lvl2' in name:
        mass = 6E-6 / 8.0 / 8.0
    
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
    else:
        center = np.array([200., 200., 200.])

    return mass, center

def _bar_prop_one_chunk(prefix_in_bar, prefix_phase_space, name, chunk_idx):
    # read in the relevant files
    fname_in_bar = prefix_in_bar + 'in_bar_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5_in_bar = h5.File(fname_in_bar, mode='r')

    fname_phase_space = prefix_phase_space + 'phase_space_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5_phase_space = h5.File(fname_phase_space, mode='r')

    # pull out in_bar from file
    in_bar = np.array(h5_in_bar['in_bar'])
    idx_list = np.array(h5_in_bar['idx_list'])
    
    bar_prop_list = {}
    for key in ['tlist', 'bar_frac', 'Rbar', 'Mbar', 'Lzbar', 'Izbar']:
        bar_prop_list[key] = np.zeros(len(idx_list))
    
    print(len(idx_list))

    # get mass and center from name
    mass, center = get_mass_and_center_from_name(name)

    # load phase space properties
    # load disk particles
    pos_tot = np.array(h5_phase_space['PartType2/Coordinates']).swapaxes(0, 1)
    vel_tot = np.array(h5_phase_space['PartType2/Velocities']).swapaxes(0, 1)

    # load bulge particles
    pos_tot = np.concatenate((pos_tot, np.array(h5_phase_space['PartType3/Coordinates']).swapaxes(0, 1) ))
    vel_tot = np.concatenate((vel_tot, np.array(h5_phase_space['PartType3/Velocities']).swapaxes(0, 1) ))

    # load star particles (if they exist)
    if 'PartType4' in h5_phase_space.keys():
        pos_tot = np.concatenate((pos_tot, np.array(h5_phase_space['PartType4/Coordinates']).swapaxes(0, 1) ))
        vel_tot = np.concatenate((vel_tot, np.array(h5_phase_space['PartType4/Velocities']).swapaxes(0, 1) ))

    pos_tot = pos_tot - center
    tlist = np.array(h5_phase_space['Time'])


    # loop over snapshots
    Rlist = {}
    N_inbar = np.zeros(len(idx_list))
    N_disk = np.zeros(len(idx_list))
    for j,idx in enumerate(idx_list):
        # pull poss, vel from current snapshot
        pos = pos_tot[:,idx,:]
        vel = vel_tot[:,idx,:]
        key = in_bar[idx]

        # separate into particles in the bar
        pos_inbar = pos[key]
        vel_inbar = vel[key]

        # count number in bar and number in disk
        N_inbar[j] = len(np.where(key)[0])
        N_disk[j] = len(key)

        # compute mass and lz of bar
        Mbar = mass * len(pos_inbar)
        Lzbar = mass * np.sum(np.cross(pos_inbar, vel_inbar)[:,2])

        Rlist[idx] = np.linalg.norm(pos_inbar[:,:2], axis=1)
        Izbar = mass * np.sum(Rlist[idx]**2)

        # output
        bar_prop_list['tlist'][j] = tlist[idx]
        bar_prop_list['Mbar'][j]  = Mbar
        bar_prop_list['Lzbar'][j] = Lzbar
        bar_prop_list['Izbar'][j] = Izbar

    
    # close h5 files
    h5_in_bar.close()
    h5_phase_space.close()

    return bar_prop_list, Rlist, (N_inbar, N_disk)

def process_bar_prop_out(bar_prop_out):
    # get total number of snapshots
    nsnap = len(bar_prop_out[0][0]['tlist'])
    nchunk = len(bar_prop_out)
    
    bar_prop = {}
    for key in ['tlist', 'bar_frac', 'Rbar', 'Mbar', 'Lzbar', 'Izbar']:
        bar_prop[key] = np.zeros(nsnap)

    N_inbar = np.zeros(nsnap)
    N_disk = np.zeros(nsnap)

    # setup Rlist
    Rlist = {}
    for j in range(nsnap):
        Rlist[j] = np.array([])

    for i in range(nchunk):
        bar_prop_i = bar_prop_out[i][0]
        
        # add the Mbar and Lzbar
        bar_prop['Mbar']  += bar_prop_i['Mbar']
        bar_prop['Lzbar'] += bar_prop_i['Lzbar']
        bar_prop['Izbar'] += bar_prop_i['Izbar']

        # concat the Rlist
        for j in range(nsnap):
            Rlist_j = bar_prop_out[i][1][j]
            if len(Rlist_j) > 0:
                Rlist[j] = np.concatenate((Rlist[j], Rlist_j))
        
        # add the number in bar and number in disk
        N_inbar += bar_prop_out[i][2][0]
        N_disk += bar_prop_out[i][2][1]
    
    # get Rbar for each snap
    Rbar = []
    for j in range(nsnap):
        if len(Rlist[j] > 100):
            Rbar_j = np.percentile(Rlist[j], 99)
        else:
            Rbar_j = 0.0
        
        Rbar.append(Rbar_j)
    
    bar_prop['Rbar'] = np.array(Rbar)

    # compute disk fraction
    bar_prop['bar_frac'] = N_inbar / N_disk
    bar_prop['tlist'] = bar_prop_out[0][0]['tlist'] # tlist

    return bar_prop

def get_other_output(prefix_in_bar, name):
    fin = prefix_in_bar + 'in_bar_' + name + '.0.hdf5'
    h5in = h5.File(fin, mode='r')

    tlist = np.array(h5in['tlist'])
    idx_list = np.array(h5in['idx_list'])
    bar_angle = np.array(h5in['bar_angle'])

    return tlist, idx_list, bar_angle

def run(name, nproc, basepath = '/n/home01/abeane/starbar/plots/bar_orbits/data/'):
    prefix_in_bar = '../in_bar/data/in_bar_' + name + '/'
    prefix_phase_space = '/n/home01/abeane/starbar/plots/phase_space/data/' + name + '/'

    nchunk = len(glob.glob(prefix_in_bar+'/in_bar_'+name+'.*.hdf5'))

    # compute bar properties for each chunk
    bar_prop_out = Parallel(n_jobs=nproc)(delayed(_bar_prop_one_chunk)(prefix_in_bar, prefix_phase_space, name, i) for i in tqdm(range(nchunk)))
    
    # process the bar output from each chunk
    bar_prop = process_bar_prop_out(bar_prop_out)

    # get other output from in bar
    tlist, idx_list, bar_angle = get_other_output(prefix_in_bar, name)

    # output
    prefix_out = 'data/'
    fout = prefix_out + 'bar_prop_' + name + '.hdf5'
    h5out = h5.File(fout, mode='w')

    grp = h5out.create_group('bar_prop')
    for key in bar_prop.keys():
        grp.create_dataset(key, data=bar_prop[key])
    h5out.create_dataset('tlist', data=tlist)
    h5out.create_dataset('idx_list', data=idx_list)
    h5out.create_dataset('bar_angle', data=bar_angle)

    h5out.close()

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
