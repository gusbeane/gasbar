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

in_bar_path = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/analysis/in_bar/data/'

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

def compute_fourier_component(path, snapnum, Rmax, nbins=60, in_bar_disk=None, in_bar_halo=None, logspace=False, center=None):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True, 
                            parttype=[1, 2, 3, 4], fields=['Coordinates', 'Masses', 'Potential', 'ParticleIDs'])
    except:
        print("unable to load path:"+path, " snapnum: ", snapnum)
        return None
    
    center = sn.part1.pos.value[np.argmin(sn.part1.pot)]

#     firstpart = True
#     for i, npart in enumerate(sn.NumPart_Total):
#         if i not in [2, 3]:
#             continue

#         if npart == 0:
#             continue

#         part = getattr(sn, 'part'+str(i))

#         # compute the center of mass
#         this_mass = sn.MassTable[i].value
#         this_pos = part.pos.value

#         if center is not None:
#             this_pos = np.subtract(this_pos, center)

#         # if mass is zero, then we need to load each individual mass
#         if this_mass == 0:
#             this_mass = part.mass
#         else:
#             this_mass = np.full(npart, this_mass)

#         # now concatenate if needed
#         if firstpart:
#             mass = np.copy(this_mass)
#             pos = np.copy(this_pos)
#             firstpart = False
#         else:
#             mass = np.concatenate((mass, this_mass))
#             pos = np.concatenate((pos, this_pos))

    pos = sn.part2.pos.value
    pos = pos - center
    mass = np.full(sn.NumPart_Total[2], sn.MassTable[2].value)
    Nbefore = len(pos)
    #pos = pos[np.logical_not(in_bar_disk)]
    #mass = mass[np.logical_not(in_bar_disk)]
    Nafter = len(pos)
    
    # do for disk
    A0, _ = fourier_component(pos, mass, 0, Rmax)
    A1r, A1i = fourier_component(pos, mass, 1, Rmax)
    A2r, A2i = fourier_component(pos, mass, 2, Rmax)

    # now do for halo
    pos = sn.part1.pos.value
    if center is not None:
        pos = pos - center
    mass = np.full(sn.NumPart_Total[1], sn.MassTable[1].value)
    
    
    Nbefore = len(pos)
    pos = pos[np.argsort(sn.part1.id)]
    
    pos = pos[np.logical_not(in_bar_halo)]
    mass = mass[np.logical_not(in_bar_halo)]
    Nafter = len(pos)

    print('Nbefore:', Nbefore, 'Nafter:', Nafter)

    A0_h, _ = fourier_component(pos, mass, 0, Rmax)
    A1r_h, A1i_h = fourier_component(pos, mass, 1, Rmax)
    A2r_h, A2i_h = fourier_component(pos, mass, 2, Rmax)

    time = sn.Time.value

    out = {}
    out['A0'] = A0
    out['A1r'] = A1r
    out['A1i'] = A1i
    out['A2r'] = A2r
    out['A2i'] = A2i

    out['A0_h'] = A0_h
    out['A1r_h'] = A1r_h
    out['A1i_h'] = A1i_h
    out['A2r_h'] = A2r_h
    out['A2i_h'] = A2i_h

    out['time'] = time

    return out

def concat_files(outs, indices, fout):
    h5out = h5.File(fout, mode='w')
    time_list = []

    A0_list = []
    A1r_list = []
    A1i_list = []
    A2r_list = []
    A2i_list = []
    
    A0_h_list = []
    A1r_h_list = []
    A1i_h_list = []
    A2r_h_list = []
    A2i_h_list = []

    for t, idx in zip(outs, indices):
        time_list.append(t['time'])

        A0_list.append(t['A0'])
        A1r_list.append(t['A1r'])
        A1i_list.append(t['A1i'])
        A2r_list.append(t['A2r'])
        A2i_list.append(t['A2i'])

        A0_h_list.append(t['A0_h'])
        A1r_h_list.append(t['A1r_h'])
        A1i_h_list.append(t['A1i_h'])
        A2r_h_list.append(t['A2r_h'])
        A2i_h_list.append(t['A2i_h'])
        
    h5out.create_dataset('time', data=time_list)

    h5out.create_dataset('A0', data=A0_list)
    h5out.create_dataset('A1r', data=A1r_list)
    h5out.create_dataset('A1i', data=A1i_list)
    h5out.create_dataset('A2r', data=A2r_list)
    h5out.create_dataset('A2i', data=A2i_list)

    h5out.create_dataset('A0_h', data=A0_h_list)
    h5out.create_dataset('A1r_h', data=A1r_h_list)
    h5out.create_dataset('A1i_h', data=A1i_h_list)
    h5out.create_dataset('A2r_h', data=A2r_h_list)
    h5out.create_dataset('A2i_h', data=A2i_h_list)

    h5out.close()

    return None

def read_in_bar(name, nchunk=256):
    fname_base = in_bar_path+'in_bar_'+name+'/in_bar_'+name+'.'
    
    out = {}
    out['PartType1'] = {}
    out['PartType1']['in_bar'] = []
    
    out['PartType2'] = {}
    out['PartType2']['in_bar'] = []
    
    for i in tqdm(range(nchunk)):
        fname = fname_base + str(i) + '.hdf5'
        t = h5.File(fname, mode='r')
        
        out['PartType1']['in_bar'].append(t['PartType1']['in_bar'][:])
        out['PartType2']['in_bar'].append(t['PartType2']['in_bar'][:])
        t.close()
        
    out['PartType1']['in_bar'] = np.concatenate(out['PartType1']['in_bar'], axis=1)
    out['PartType2']['in_bar'] = np.concatenate(out['PartType2']['in_bar'], axis=1)
    
    return out

def run(path, name, nsnap):
    fout = 'data/fourier_' + name + '.hdf5'

    Rmax = 4.0

    if 'Nbody' in name:
        center = None
    else:
        center = np.array([200, 200, 200])
    
    in_bar = read_in_bar(name)

    indices = np.arange(nsnap)
    outs = Parallel(n_jobs=nproc) (delayed(compute_fourier_component)(path, int(idx), Rmax, in_bar_disk=in_bar['PartType2']['in_bar'][idx], in_bar_halo = in_bar['PartType1']['in_bar'][idx], center=center) for idx in tqdm(indices))

    concat_files(outs, indices, fout)
    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

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
