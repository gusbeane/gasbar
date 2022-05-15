import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import os

from joblib import Parallel, delayed

import agama
agama.setUnits(mass=1E10, length=1, velocity=1)

def read_snap(idx, name, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs'],
              basepath = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'):
    
    fname = basepath + name + '/output'
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

def get_pos_vel_mass_halo_bulge(sn, center=np.array([0., 0., 0.])):
    pos_halo = sn.part1.pos.value[np.argsort(sn.part1.id)]
    vel_halo = sn.part1.vel.value[np.argsort(sn.part1.id)]
    mass_halo = np.full(sn.NumPart_Total[1], sn.MassTable[1])

    pos_disk = sn.part2.pos.value[np.argsort(sn.part2.id)]
    vel_disk = sn.part2.vel.value[np.argsort(sn.part2.id)]
    
    pos_blge = sn.part3.pos.value[np.argsort(sn.part3.id)]
    vel_blge = sn.part3.vel.value[np.argsort(sn.part3.id)]
    
    if sn.NumPart_Total[0] > 0:
        pos_gas = sn.part0.pos.value
        vel_gas = sn.part0.vel.value
        mass_gas = sn.part0.mass.value
        
        if sn.NumPart_Total[4]>0:
            pos_star = sn.part4.pos.value
            vel_star = sn.part4.vel.value
            mass_star = sn.part4.mass.value

    pos_bar = np.concatenate((pos_disk, pos_blge))
    vel_bar = np.concatenate((vel_disk, vel_blge))
    mass_bar = np.concatenate((np.full(sn.NumPart_Total[2], sn.MassTable[2]), \
                               np.full(sn.NumPart_Total[3], sn.MassTable[3])))

    if sn.NumPart_Total[0] > 0:
        pos_bar = np.concatenate((pos_bar, pos_gas))
        vel_bar = np.concatenate((vel_bar, vel_gas))
        mass_bar = np.concatenate((mass_bar, mass_gas))
        if sn.NumPart_Total[4] > 0:
            pos_bar = np.concatenate((pos_bar, pos_star))
            vel_bar = np.concatenate((vel_bar, vel_star))
            mass_bar = np.concatenate((mass_bar, mass_star))

    pos_bar -= center
    pos_halo -= center
            
    return pos_bar, vel_bar, mass_bar, pos_halo, vel_halo, mass_halo

def _run_chunk(name, idx, prefix, center):
    sn = read_snap(idx, name.replace('-lvl', '/lvl'), parttype=None)
    agama.setUnits(mass=1E10, length=1, velocity=1)

    pos_bar, vel_bar, mass_bar, pos_halo, vel_halo, mass_halo = get_pos_vel_mass_halo_bulge(sn, center=center)

    pot_halo = agama.Potential(type="Multipole", particles=(pos_halo, mass_halo),
                           symmetry='a', gridsizeR=20, lmax=2)

    pot_bar  = agama.Potential(type="CylSpline", particles=(pos_bar, mass_bar), 
                          symmetry='a', gridsizer=20, gridsizez=20,
                          mmax=0, Rmin=0.2, Rmax=50, Zmin=0.02, Zmax=10)

    pot_tot = agama.Potential(pot_halo, pot_bar)

    fout = prefix + 'pot_' + name + '.' + str(idx) + '.txt'

    pot_tot.export(fout)

def preprocess_center(name):
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
        firstkey=150
        indices = np.arange(nsnap)
        # indices_analyze = np.arange(500, 1100, 20)
    else:
        center = np.array([200, 200, 200])
        firstkey=0
        indices = np.arange(nsnap)

    return center, firstkey, indices

def run(path, name, nsnap, nproc, phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data/'):
    prefix = 'data/pot_' + name +'/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # get some preliminary variables
    center, firstkey, indices = preprocess_center(name)
    
    # do standard fourier and bar angle stuff
    _ = Parallel(n_jobs=nproc) (delayed(_run_chunk)(name, i, prefix, center) for i in tqdm(range(nsnap)))
        
    # for i in tqdm(range(nsnap)):
    #    print(i)
    #    _run_chunk(name, i, prefix, center)

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

        out = run(path, name, nsnap, nproc)
    else:
        for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
            out = run(path, name, nsnap, nproc)
