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
import shutil

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
   
    print('starting snap_idx', snap_idx)

    # read in the relevant files
    pos_bar = np.array([]).reshape((0, 3))
    pos_notbar = np.array([]).reshape((0, 3))
    mass_disk, center = get_mass_and_center_from_name(name)

    in_bar_pt2 = []
    in_bar_pt3 = []
    in_bar_pt4 = []

    # load snapshot
    sn = read_snap(path, snap_idx, parttype=None, 
                   fields=['Coordinates', 'Masses', 'Softenings', 'ParticleIDs'])
   
    pos_pt4 = []
    for i in tqdm(range(nchunk)):
        fname_in_bar = prefix_in_bar + 'in_bar_' + name + '.' + str(i) + '.hdf5'
        h5_in_bar = h5.File(fname_in_bar, mode='r')

        in_bar_pt2.append(h5_in_bar['PartType2/in_bar'][snap_idx])
        in_bar_pt3.append(h5_in_bar['PartType3/in_bar'][snap_idx])
        if 'PartType4' in h5_in_bar.keys():
            in_bar_pt4.append(h5_in_bar['PartType4/in_bar'][snap_idx])

        fname_phase_space = prefix_phase_space + '/phase_space_' + name + '.' + str(i) + '.hdf5'
        h5_ps = h5.File(fname_phase_space, mode='r')

        pos_pt4.append( h5_ps['PartType4/Coordinates'][:,snap_idx,:] )

        h5_in_bar.close()

    pos_pt4 = np.concatenate(pos_pt4)

    in_bar_pt2 = np.concatenate(in_bar_pt2)
    in_bar_pt3 = np.concatenate(in_bar_pt3)
    if len(in_bar_pt4) > 0:
        in_bar_pt4 = np.concatenate(in_bar_pt4)

    pos_pt2 = sn.part2.pos.value[ np.argsort(sn.part2.id) ] - center
    pos_pt3 = sn.part3.pos.value[ np.argsort(sn.part3.id) ] - center
    #if sn.NumPart_Total[4] > 0:
    #    pos_pt4 = sn.part4.pos.value[ np.argsort(sn.part4.id) ] - center

    pos_bar = []
    pos_bar.append(pos_pt2[in_bar_pt2])
    pos_bar.append(pos_pt3[in_bar_pt3])
    if sn.NumPart_Total[4]>0:
        pos_bar.append(pos_pt4[in_bar_pt4])
    pos_bar = np.concatenate(pos_bar)

    out_bar_pt2 = np.logical_not(in_bar_pt2)
    out_bar_pt3 = np.logical_not(in_bar_pt3)
    out_bar_pt4 = np.logical_not(in_bar_pt4)

    pos_notbar = []
    pos_notbar.append(pos_pt2[out_bar_pt2])
    pos_notbar.append(pos_pt3[out_bar_pt3])
    if sn.NumPart_Total[4] > 0:
        pos_notbar.append(pos_pt4[out_bar_pt4])
    pos_notbar = np.concatenate(pos_notbar)

    bar_mass = []
    bar_mass.append(np.full(sn.NumPart_Total[2], sn.MassTable[2].value))
    bar_mass.append(np.full(sn.NumPart_Total[3], sn.MassTable[3].value))
    if sn.NumPart_Total[4] > 0.0:
        bar_mass.append(sn.part4.mass.value)
    bar_mass = np.concatenate(bar_mass)

    bar_soft = sn.part2.soft[0]

    print('a', snap_idx)

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

    print('b', snap_idx)
    
    #fout = path_out + 'torques_' + name + '.' + str(snap_idx) + '.hdf5'
    fout_tmp = '/tmp/torques_' + name + '.' + str(snap_idx) + '.hdf5'
    h5out = h5.File(fout_tmp, mode='w')

    print('c', snap_idx)

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

    print('d', snap_idx)

    # h5out.create_dataset("acc_bulge", data=acc_out[3])
    # h5out.create_dataset("pos_bulge", data=pos_out[3])

    # if 4 in acc_out.keys():
        # h5out.create_dataset("acc_star", data=acc_out[4])
        # h5out.create_dataset("pos_star", data=pos_out[4])
    
    h5out.close()

    shutil.copy(fout_tmp, path_out)
    os.remove(fout_tmp)

    print('e', snap_idx)

    return None

def run(path, name, snap_idx, basepath = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/bar_orbits/data/'):
    prefix_in_bar = '../in_bar/data/in_bar_' + name + '/'
    prefix_phase_space = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/phase_space/data/' + name + '/'

    path_out = 'data/' + 'torques_' + name + '/'
    try:
        os.mkdir(path_out)
    except:
        pass

    nchunk = len(glob.glob(prefix_in_bar+'/in_bar_'+name+'.*.hdf5'))

    # compute bar properties for each chunk
    # _ = Parallel(n_jobs=nproc)(delayed(_torque_one_snap)(path, prefix_in_bar, prefix_phase_space, name, nchunk, i, path_out) for i in tqdm(range(nsnap)))
    
    #for i in tqdm(range(nsnap)):
    #    _torque_one_snap(path, prefix_in_bar, prefix_phase_space, name, nchunk, i, path_out)

    _torque_one_snap(path, prefix_in_bar, prefix_phase_space, name, nchunk, snap_idx, path_out)

    return None        

if __name__ == '__main__':
    basepath = '../../runs/'
    
    name_prefix = sys.argv[1]
    lvl = sys.argv[2]
    snap_idx = int(sys.argv[3])

    path = basepath + name_prefix + '/' + lvl
    name = name_prefix + '-' + lvl

    out = run(path, name, snap_idx)

