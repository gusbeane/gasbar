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
def fourier_component(pos, mass, m, Rmin, Rmax, nbins=20, logspace=True):
    if logspace:
        bins = np.linspace(np.log10(Rmin), np.log10(Rmax), nbins+1)
        bins = np.power(10., bins)
    else:
        bins = np.linspace(Rmin, Rmax, nbins+1)
    
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

def compute_fourier_component(path, snapnum, Rmin=0.0, Rmax=30.0, nbins=60, logspace=False, center=None):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True, 
                            parttype=[2, 3, 4], fields=['Coordinates', 'Masses'])
    except:
        print("unable to load path:"+path, " snapnum: ", snapnum)
        return None
    
    firstpart = True
    for i, npart in enumerate(sn.NumPart_Total):
        if i not in [2, 3, 4]:
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

    Rlist, A0, _ = fourier_component(pos, mass, 0, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A1r, A1i = fourier_component(pos, mass, 1, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A2r, A2i = fourier_component(pos, mass, 2, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A3r, A3i = fourier_component(pos, mass, 3, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A4r, A4i = fourier_component(pos, mass, 4, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A5r, A5i = fourier_component(pos, mass, 5, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A6r, A6i = fourier_component(pos, mass, 6, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A7r, A7i = fourier_component(pos, mass, 7, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A8r, A8i = fourier_component(pos, mass, 8, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A9r, A9i = fourier_component(pos, mass, 9, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A10r, A10i = fourier_component(pos, mass, 10, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    
    time = sn.Time.as_unit(arepo.u.d).value * u.d
    time = time.to_value(u.Myr)

    out = {}
    out['Rlist'] = Rlist
    out['A0'] = A0
    out['A1r'], out['A1i'] = A1r, A1i
    out['A2r'], out['A2i'] = A2r, A2i
    out['A3r'], out['A3i'] = A3r, A3i
    out['A4r'], out['A4i'] = A4r, A4i
    out['A5r'], out['A5i'] = A5r, A5i
    out['A6r'], out['A6i'] = A6r, A6i
    out['A7r'], out['A7i'] = A7r, A7i
    out['A8r'], out['A8i'] = A8r, A8i
    out['A9r'], out['A9i'] = A9r, A9i
    out['A10r'], out['A10i'] = A10r, A10i
    out['time'] = time

    return out

def concat_files(outs, indices, fout):
    h5out = h5.File(fout, mode='w')
    time_list = []

    for t, idx in zip(outs, indices):
        snap = h5out.create_group('snapshot_'+"{:03d}".format(idx))

        for key in ['Rlist', 'A0', 'A1r', 'A1i',
                    'A2r','A2i', 'A3r','A3i',
                    'A4r','A4i', 'A5r','A5i',
                    'A6r','A6i', 'A7r','A7i',
                    'A8r','A8i', 'A9r','A9i', 'A10r', 'A10i']:
            snap.create_dataset(key, data=t[key])
        time_list.append(t['time'])

    h5out.create_dataset('time', data=time_list)
    h5out.close()

    return None

def run(path, name, nsnap):
    fout = 'data/fourier_' + name + '.hdf5'

    # dont remake something already made
    if os.path.exists(fout):
        return None

    if 'Nbody' in name:
        center = None
    else:
        center = np.array([200, 200, 200])

    indices = np.arange(nsnap)
    outs = Parallel(n_jobs=nproc) (delayed(compute_fourier_component)(path, int(idx), center=center) for idx in tqdm(indices))

    concat_files(outs, indices, fout)
    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    fid_dP = 'fRpoly'
    fid_dP_c1 = 'fRpoly-Rcore1.0'
    fid_dP2_c1 = 'fRpoly2-Rcore1.0'
    fid_dP_c1_rx = 'fRpoly-Rcore1.0-relax'
    fid_dP_c1_bG = 'fRpoly-Rcore1.0-barGas'
    fid_dP_c1_bG1 = 'fRpoly-Rcore1.0-barGas1.0'
    fid_dP_c1_bG2 = 'fRpoly-Rcore1.0-barGas2.0'
    fid_dP_c1_sp = 'fRpoly-Rcore1.0-spring'
    fid_dP_c1_MB = 'fRpoly-Rcore1.0-MB0.004'
    fid_dP_c1_rB = 'fRpoly-Rcore1.0-ringBug'
    fid_dP_c1_h = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel160.0'
    fid_dP_c1_h_v140 = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel140.0'

    phv = 'phantom-vacuum'
    phgv = 'phantom-vacuum-grav'

    phgvS1 = 'phantom-vacuum-Sg10-Rc4.0'
    phgvS2 = 'phantom-vacuum-Sg20-Rc4.0'
    phgS1 = 'phantom-Sg10-Rc4.0'

    pair_list = [#(fid_dP, 'lvl5'), (fid_dP, 'lvl4'), #(fid_dP, 'lvl3'),
                 (Nbody, 'lvl5'), (Nbody, 'lvl4'), (Nbody, 'lvl3'),
                 (fid_dP_c1, 'lvl5'), (fid_dP_c1, 'lvl4'), (fid_dP_c1, 'lvl3'),
                 (fid_dP2_c1, 'lvl5'), (fid_dP2_c1, 'lvl4'), (fid_dP2_c1, 'lvl3'),
                 (fid_dP_c1_bG2, 'lvl5'), (fid_dP_c1_bG2, 'lvl4'), (fid_dP_c1_bG2, 'lvl3'),
                 (fid_dP_c1_sp, 'lvl5'), (fid_dP_c1_sp, 'lvl4'), (fid_dP_c1_sp, 'lvl3'),
                 (fid_dP_c1_MB, 'lvl5'), (fid_dP_c1_MB, 'lvl4'), (fid_dP_c1_MB, 'lvl3'),
                 (phv, 'lvl3'),
                 (phgv, 'lvl3'),
                 (phgvS1, 'lvl3'),
                 (phgvS2, 'lvl3'),
                 (phgS1, 'lvl3')]
                 # (fid_dP_c1_bG, 'lvl5'), (fid_dP_c1_bG, 'lvl4'),# (fid_dP_c1_bG, 'lvl3'),
                 # (fid_dP_c1_bG1, 'lvl5'),# (fid_dP_c1_bG, 'lvl4'),# (fid_dP_c1_bG, 'lvl3'),
                 # (fid_dP_c1_rB, 'lvl5'), (fid_dP_c1_rB, 'lvl4'), (fid_dP_c1_rB, 'lvl3'),
                 # (fid_dP_c1_h, 'lvl5'), (fid_dP_c1_h, 'lvl4'), #(fid_dP_c1_h, 'lvl3'),
                 # (fid_dP_c1_h_v140, 'lvl5'), (fid_dP_c1_h_v140, 'lvl4')] #(fid_dP_c1_h, 'lvl3')]

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
