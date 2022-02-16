import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import h5py as h5
import glob
import os
from numba import njit

from joblib import Parallel, delayed

@njit
def fourier_component(pos, mass, m, Rmax):
    Am_r = 0.0
    Am_i = 0.0
    Rmaxsq = Rmax*Rmax

    Npart = len(pos)
    for i in range(Npart):
        Rsq = pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1] + pos[i][2]*pos[i][2]
        
        if Rsq < Rmaxsq:
            phi = np.arctan2(pos[i][1], pos[i][0])
            Am_r += mass[i] * np.cos(m*phi)
            Am_i += mass[i] * np.sin(m*phi)

    return Am_r, Am_i

def compute_fourier_component(path, snapnum, Rmax, nbins=60, logspace=False, center=None):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True, 
                            parttype=[1, 2, 3, 4], fields=['Coordinates', 'Masses'])
    except:
        print("unable to load path:"+path, " snapnum: ", snapnum)
        return None
    
    firstpart = True
    for i, npart in enumerate(sn.NumPart_Total):
        if i not in [2, 3]:
            continue

        if npart == 0:
            continue

        part = getattr(sn, 'part'+str(i))

        # compute the center of mass
        this_mass = sn.MassTable[i].as_unit(arepo.u.msol).value
        this_pos = part.pos.as_unit(arepo.u.kpc).value

        if center is not None:
            this_pos = np.subtract(this_pos, center)

        # if mass is zero, then we need to load each individual mass
        if this_mass == 0:
            this_mass = part.mass.as_unit(arepo.u.msol).value
        else:
            this_mass = np.full(npart, this_mass)

        # now concatenate if needed
        if firstpart:
            mass = np.copy(this_mass)
            pos = np.copy(this_pos)
            firstpart = False
        else:
            mass = np.concatenate((mass, this_mass))
            pos = np.concatenate((pos, this_pos))

    # do for disk
    A0, _ = fourier_component(pos, mass, 0, Rmax)
    A2r, A2i = fourier_component(pos, mass, 2, Rmax)

    # now do for halo
    pos = sn.part1.pos.value
    if center is not None:
        pos = pos - center
    mass = np.full(sn.NumPart_Total[1], sn.MassTable[1].value)

    A0_h, _ = fourier_component(pos, mass, 0, Rmax)
    A2r_h, A2i_h = fourier_component(pos, mass, 2, Rmax)

    time = sn.Time.value

    out = {}
    out['A0'] = A0
    out['A2r'] = A2r
    out['A2i'] = A2i

    out['A0_h'] = A0_h
    out['A2r_h'] = A2r_h
    out['A2i_h'] = A2i_h

    out['time'] = time

    return out

def concat_files(outs, indices, fout):
    h5out = h5.File(fout, mode='w')
    time_list = []

    A0_list = []
    A2r_list = []
    A2i_list = []
    
    A0_h_list = []
    A2r_h_list = []
    A2i_h_list = []

    for t, idx in zip(outs, indices):
        time_list.append(t['time'])

        A0_list.append(t['A0'])
        A2r_list.append(t['A2r'])
        A2i_list.append(t['A2i'])

        A0_h_list.append(t['A0_h'])
        A2r_h_list.append(t['A2r_h'])
        A2i_h_list.append(t['A2i_h'])
        
    h5out.create_dataset('time', data=time_list)

    h5out.create_dataset('A0', data=A0_list)
    h5out.create_dataset('A2r', data=A2r_list)
    h5out.create_dataset('A2i', data=A2i_list)

    h5out.create_dataset('A0_h', data=A0_h_list)
    h5out.create_dataset('A2r_h', data=A2r_h_list)
    h5out.create_dataset('A2i_h', data=A2i_h_list)

    h5out.close()

    return None

def run(path, name, nsnap):
    fout = 'data/fourier_' + name + '.hdf5'

    Rmax = 4.0

    # dont remake something already made
    if os.path.exists(fout):
        return None

    if 'Nbody' in name:
        center = None
    else:
        center = np.array([200, 200, 200])

    indices = np.arange(nsnap)
    outs = Parallel(n_jobs=nproc) (delayed(compute_fourier_component)(path, int(idx), Rmax, center=center) for idx in tqdm(indices))

    concat_files(outs, indices, fout)
    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl4'), (Nbody, 'lvl3'), (Nbody, 'lvl2'),
                 (phgvS2Rc35, 'lvl4'), (phgvS2Rc35, 'lvl3')]

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
