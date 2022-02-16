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

from pyMND.forcetree import construct_tree, force_treeevaluate_loop

from joblib import Parallel, delayed

def read_snap(path, idx, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs']):
    
    fname = path + '/output'
    
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

@njit
def my_mult(mass, vel):
    out = np.zeros((len(mass), 3))
    for i in range(len(mass)):
        out[i][0] = mass[i] * vel[i][0] 
        out[i][1] = mass[i] * vel[i][1]
        out[i][2] = mass[i] * vel[i][2]
    return out

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

        pos_tot = np.array(h5in['PartType2/Coordinates'])
        vel_tot = np.array(h5in['PartType2/Velocities'])
        ids = np.array(h5in['PartType2/ParticleIDs'])

        pos_tot = np.concatenate((pos_tot, np.array(h5in['PartType3/Coordinates'])))
        vel_tot = np.concatenate((vel_tot, np.array(h5in['PartType3/Velocities'])))
        ids = np.concatenate((ids, np.array(h5in['PartType3/ParticleIDs'])))

        if 'PartType4' in h5in.keys():
            pos_tot = np.concatenate((pos_tot, np.array(h5in['PartType4/Coordinates'])))
            vel_tot = np.concatenate((vel_tot, np.array(h5in['PartType4/Velocities'])))
            ids = np.concatenate((ids, np.array(h5in['PartType4/ParticleIDs'])))
        
        pos_tot -= center
    
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

def _torque_one_snap(path, prefix_in_bar, prefix_phase_space, name, nchunk, snap_idx, 
                     path_out, theta=0.35, maxnode_fac=5., num_threads=1, G=43018.7):
    
    # read in the relevant files
    pos_bar = np.array([]).reshape((0, 3))
    pos_notbar = np.array([]).reshape((0, 3))
    mass_disk, center = get_mass_and_center_from_name(name)

    for i in range(nchunk):
        fname_in_bar = prefix_in_bar + 'in_bar_' + name + '.' + str(i) + '.hdf5'
        h5_in_bar = h5.File(fname_in_bar, mode='r')

        fname_phase_space = prefix_phase_space + 'phase_space_' + name + '.' + str(i) + '.hdf5'
        h5_phase_space = h5.File(fname_phase_space, mode='r')

        # pull out in_bar from file
        in_bar = np.array(h5_in_bar['in_bar'])
        # idx_list = np.array(h5_in_bar['idx_list'])
    
        # get mass and center from name

        # load phase space properties
        pos_tot = np.array(h5_phase_space['PartType2/Coordinates'][snap_idx])
        pos_tot = np.concatenate((pos_tot, np.array(h5_phase_space['PartType3/Coordinates'][snap_idx])))
        if 'PartType4' in h5_phase_space.keys():
            pos_tot = np.concatenate((pos_tot, np.array(h5_phase_space['PartType4/Coordinates'][snap_idx])))

        pos_tot -= center

        
        # vel_tot = np.array(h5_phase_space['Velocities'])
        # tlist = np.array(h5_phase_space['Time'])

        in_bar_idx = in_bar[snap_idx]
        out_bar_idx = np.logical_not(in_bar_idx)

        pos_bar = np.concatenate((pos_bar, pos_tot[in_bar_idx]))
        pos_notbar = np.concatenate((pos_notbar, pos_tot[out_bar_idx]))

        h5_in_bar.close()
        h5_phase_space.close()
    
    bar_mass = np.full(len(pos_bar), mass_disk)
    
    # load snapshot
    sn = read_snap(path, snap_idx, parttype=None, fields=['Coordinates', 'Masses', 'Softenings'])

    # bar_soft = np.full(len(pos_bar), sn.part2.soft[0]) # all disk particles have same softening
    bar_soft = sn.part2.soft[0]

    tree_bar = construct_tree(pos_bar, bar_mass, theta, bar_soft, maxnode_fac=maxnode_fac)
    acc_bar = G * np.array(force_treeevaluate_loop(pos_bar, tree_bar, num_threads=num_threads))
    acc_notbar = G * np.array(force_treeevaluate_loop(pos_notbar, tree_bar, num_threads=num_threads))

    acc_out = {}
    pos_out = {}
    # now loop through particle types
    # for i in range(6):
    if sn.NumPart_Total[0] > 0:
        pos_gas = sn.part0.pos.value - center
        acc_gas = G * np.array(force_treeevaluate_loop(pos_gas, tree_bar, num_threads=num_threads))

    pos_halo = sn.part1.pos.value - center
    acc_halo = G * np.array(force_treeevaluate_loop(pos_halo, tree_bar, num_threads=num_threads))

    
    fout = path_out + 'torques_' + name + '.' + str(snap_idx) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    h5out.create_dataset("acc_bar", data=acc_bar)
    h5out.create_dataset("pos_bar", data=pos_bar)

    h5out.create_dataset("acc_notbar", data=acc_notbar)
    h5out.create_dataset("pos_notbar", data=pos_notbar)

    if sn.NumPart_Total[0] > 0:
        h5out.create_dataset("acc_gas", data=acc_gas)
        h5out.create_dataset("pos_gas", data=pos_gas)
        h5out.create_dataset("gas_mass", data=sn.part0.mass.value)
    
    h5out.create_dataset("acc_halo", data=acc_halo)
    h5out.create_dataset("pos_halo", data=pos_halo)
    
    g = h5out.create_group("total_torques")
    g.attrs.create("bar", sn.MassTable[2] * np.sum(np.cross(pos_bar, acc_bar), axis=0))
    g.attrs.create("not_bar", sn.MassTable[2] * np.nansum(np.cross(pos_notbar, acc_notbar), axis=0))
    g.attrs.create("halo", sn.MassTable[1] * np.sum(np.cross(pos_halo, acc_halo), axis=0))
    if sn.NumPart_Total[0] > 0:
        acc_gas = my_mult(sn.part0.mass.value, acc_gas)
        g.attrs.create("gas", np.sum(np.cross(pos_gas, acc_gas), axis=0))
    
    g = h5out.create_group("parameters")
    g.attrs.create("Time", sn.Time.value)
    g.attrs.create("MassTable", sn.MassTable)


    # h5out.create_dataset("acc_bulge", data=acc_out[3])
    # h5out.create_dataset("pos_bulge", data=pos_out[3])

    # if 4 in acc_out.keys():
        # h5out.create_dataset("acc_star", data=acc_out[4])
        # h5out.create_dataset("pos_star", data=pos_out[4])
    
    h5out.close()

    return None


def process_bar_prop_out(bar_prop_out):
    # get total number of snapshots
    nsnap = len(bar_prop_out[0][0])
    nchunk = len(bar_prop_out)
    
    bar_prop = np.zeros((nsnap, 5))
    N_inbar = np.zeros(nsnap)
    N_disk = np.zeros(nsnap)

    # setup Rlist
    Rlist = {}
    for j in range(nsnap):
        Rlist[j] = np.array([])

    for i in range(nchunk):
        bar_prop_i = bar_prop_out[i][0]
        
        # add the Mbar and Lzbar
        bar_prop[:,3] += bar_prop_i[:,3]
        bar_prop[:,4] += bar_prop_i[:,4]

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
    
    bar_prop[:,2] = np.array(Rbar)

    # compute disk fraction
    bar_prop[:,1] = N_inbar / N_disk
    bar_prop[:,0] = bar_prop_out[0][0][:,0] # tlist

    return bar_prop

def get_other_output(prefix_in_bar, name):
    fin = prefix_in_bar + 'in_bar_' + name + '.0.hdf5'
    h5in = h5.File(fin, mode='r')

    tlist = np.array(h5in['tlist'])
    idx_list = np.array(h5in['idx_list'])
    bar_angle = np.array(h5in['bar_angle'])

    return tlist, idx_list, bar_angle

def run(path, name, nproc, nsnap, basepath = '/n/home01/abeane/starbar/plots/bar_orbits/data/'):
    prefix_in_bar = '../in_bar/data/in_bar_' + name + '/'
    prefix_phase_space = '/n/home01/abeane/starbar/plots/phase_space/data/' + name + '/'

    path_out = 'data/' + 'torques_' + name + '/'
    if not os.path.isdir(path_out):
        os.mkdir(path_out)

    nchunk = len(glob.glob(prefix_in_bar+'/in_bar_'+name+'.*.hdf5'))

    # compute bar properties for each chunk
    _ = Parallel(n_jobs=nproc)(delayed(_torque_one_snap)(path, prefix_in_bar, prefix_phase_space, name, nchunk, i, path_out) for i in tqdm(range(nsnap)))
    
    # process the bar output from each chunk
    # bar_prop = process_bar_prop_out(bar_prop_out)

    # get other output from in bar
    # tlist, idx_list, bar_angle = get_other_output(prefix_in_bar, name)

    # output
    # prefix_out = 'data/'
    # fout = prefix_out + 'bar_prop_' + name + '.hdf5'
    # h5out = h5.File(fout, mode='w')

    # h5out.create_dataset('bar_prop', data=bar_prop)
    # h5out.create_dataset('tlist', data=tlist)
    # h5out.create_dataset('idx_list/', data=idx_list)
    # h5out.create_dataset('bar_angle', data=bar_angle)
    # h5out.close()

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

    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    if len(sys.argv) == 3:
        i = int(sys.argv[2])
        path = path_list[i]
        name = name_list[i]
        nsnap = nsnap_list[i]

        out = run(path, name, nproc, nsnap)
    else:
        for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
            out = run(path, name, nproc, nsnap)
