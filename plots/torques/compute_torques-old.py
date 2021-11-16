import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import h5py as h5
import glob
import os
from numba import njit

from pyMND.forcetree import construct_tree, force_treeevaluate_loop

from joblib import Parallel, delayed

def compute_acc(pos, tree_pos, tree_mass, theta, soft, num_threads=1, maxnode_fac=2.):
    tree = construct_tree(tree_pos, tree_mass, theta, soft, maxnode_fac=maxnode_fac)
    acc = force_treeevaluate_loop(pos, tree, num_threads=num_threads)
    return acc

    
    Am_real = np.zeros(nbins)
    Am_imag = np.zeros(nbins)
    Rmag = np.zeros(nbins)
    N_in_bin = np.zeros(nbins)
    
    Npart = len(pos)
    for i in range(Npart):
        R = np.sqrt(pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1])
        phi = np.arctan2(pos[i][1], pos[i][0])
        
        for j in range(nbins):
            if R > bins[j] and R < bins[j+1]:
                Am_real[j] += mass[j]*np.cos(m*phi)
                Am_imag[j] += mass[j]*np.sin(m*phi)
                Rmag[j] += R
                N_in_bin[j] += 1
    
    
    for j in range(nbins):
        if N_in_bin[j] > 0:
            Rmag[j] /= N_in_bin[j]
        else:
            Rmag[j] = np.nan

    return Rmag, Am_real, Am_imag

def get_cyl(sn, pt, center=np.array([0., 0., 0.])):
    part = getattr(sn, 'part'+str(pt))
    pos = part.pos.value - center
    vel = part.vel.value
    
    R = np.linalg.norm(pos[:,:2], axis=1)
    phi = np.arctan2(pos[:,1], pos[:,0])
    z = pos[:,2]
    
    cosphi = pos[:,0]/R
    sinphi = pos[:,1]/R
    
    vR = cosphi * vel[:,0] + sinphi * vel[:,1]
    vphi = cosphi * vel[:,1] - sinphi * vel[:,0]
    vz = vel[:,2]
    
    return np.transpose([R, phi, z, vR, vphi, vz])

def compute_acc_alltypes(path, snapnum, center=np.array([0., 0., 0.]), bar_Lz=400, theta=0.35, num_threads=1):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True, 
                            fields=['Coordinates', 'Masses', 'Velocities'])
    except:
        print("unable to load path:"+path, " snapnum: ", snapnum)
        return None
   
    acc_out = {}

    # now separate into bar/not bar for disk
    cyl = get_cyl(sn, 2, center=center)
    R = cyl[:,0]
    vphi = cyl[:,4]
    bar_key = R * vphi < bar_Lz
    disk_key = np.logical_not(bar_key)

    bar_pos = sn.part2.pos.value[bar_key]

    for i, npart in enumerate(sn.NumPart_Total):
        if npart == 0:
            continue

        part = getattr(sn, 'part'+str(i))
        soft = getattr(sn.parameters, 'SofteningComovingType'+str(i))

        # compute the center of mass
        mass_val = sn.MassTable[i].value
        tree_pos = part.pos.value

        if center is not None:
            tree_pos = np.subtract(tree_pos, center)

        # if mass is zero, then we need to load each individual mass
        if mass_val == 0:
            tree_mass = part.mass.value
        else:
            tree_mass = np.full(npart, mass_val)

        if i==2:
            tree_pos_b = tree_pos[bar_key]
            tree_mass_b = tree_mass[bar_key]

            tree_pos_d = tree_pos[disk_key]
            tree_mass_d = tree_mass[disk_key]
   
            acc_b = compute_acc(bar_pos, tree_pos_b, tree_mass_b, theta, soft, num_threads=num_threads)
            acc_d = compute_acc(bar_pos, tree_pos_d, tree_mass_d, theta, soft, num_threads=num_threads)

            acc_out['acc_bar'] = acc_b
            acc_out['acc_disk'] = acc_d
        else:
            if i==0:
                name = 'acc_gas'
                maxnode_fac=20.0
            elif i==1:
                name = 'acc_halo'
                maxnode_fac=1.5
            elif i==3:
                name = 'acc_bulge'
                maxnode_fac=1.5
            elif i==4:
                name == 'acc_star'
                maxnode_fac=1.5

            print(i, name, maxnode_fac)
            acc = compute_acc(bar_pos, tree_pos, tree_mass, theta, soft, num_threads=num_threads, maxnode_fac=maxnode_fac)
            print('done')
            acc_out[name] = acc

    acc_out['time'] = sn.Time.value

    return acc_out

def concat_files(outs, indices, fout):
    h5out = h5.File(fout, mode='w')
    time_list = []

    for t, idx in zip(outs, indices):
        snap = h5out.create_group('snapshot_'+"{:03d}".format(idx))

        for key in ['acc_gas', 'acc_halo', 'acc_bar', 'acc_disk', 'acc_bulge', 'acc_star']:
            if key in t.keys():
                snap.create_dataset(key, data=t[key])
        time_list.append(t['time'])

    h5out.create_dataset('time', data=np.array(time_list))
    h5out.close()

    return None

def run(path, name, nsnap, stride=50):
    fout = 'data/torques_' + name + '.hdf5'

    # dont remake something already made
    if os.path.exists(fout):
        return None

    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
    else:
        center = np.array([200, 200, 200])

    indices = np.arange(nsnap)
    indices = indices[::stride]
    #outs = Parallel(n_jobs=nproc) (delayed(compute_fourier_component)(path, int(idx), center=center) for idx in indices)
    outs = [compute_acc_alltypes(path, int(idx), center=center, num_threads=nproc) for idx in tqdm(indices)]

    concat_files(outs, indices, fout)
    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'

    phgvS1 = 'phantom-vacuum-Sg10-Rc4.0'
    phgvS2 = 'phantom-vacuum-Sg20-Rc4.0'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'
    phgS1 = 'phantom-Sg10-Rc4.0'

    pair_list = [(Nbody, 'lvl3'),
                 (phgvS2Rc35, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                            
    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    if len(sys.argv) == 3:
        i = int(sys.argv[2])
        path = path_list[i]
        name = name_list[i]
        nsnap = nsnap_list[i]

        run(path, name, nsnap)
    else:
        for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
            run(path, name, nsnap)
