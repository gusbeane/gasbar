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
    Am_real_cum = np.zeros(nbins)
    Am_imag_cum = np.zeros(nbins)
    Rmag = np.zeros(nbins)
    N_in_bin = np.zeros(nbins)
    N_in_bin_cum = np.zeros(nbins)

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
            
            if R < bins[j+1]:
                Am_real_cum[j] += mass[j]*np.cos(m*phi)
                Am_imag_cum[j] += mass[j]*np.sin(m*phi)
                N_in_bin_cum[j] += 1
    
    
    for j in range(nbins):
        if N_in_bin[j] > 0:
            Rmag[j] /= N_in_bin[j]
        else:
            Rmag[j] = np.nan

    return Rmag, Am_real, Am_imag, Am_real_cum, Am_imag_cum

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

    Rlist, A0, _, A0_c, _ = fourier_component(pos, mass, 0, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A1r, A1i, A1r_c, A1i_c = fourier_component(pos, mass, 1, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A2r, A2i, A2r_c, A2i_c = fourier_component(pos, mass, 2, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A3r, A3i, A3r_c, A3i_c = fourier_component(pos, mass, 3, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A4r, A4i, A4r_c, A4i_c = fourier_component(pos, mass, 4, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A5r, A5i, A5r_c, A5i_c = fourier_component(pos, mass, 5, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A6r, A6i, A6r_c, A6i_c = fourier_component(pos, mass, 6, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A7r, A7i, A7r_c, A7i_c = fourier_component(pos, mass, 7, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A8r, A8i, A8r_c, A8i_c = fourier_component(pos, mass, 8, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A9r, A9i, A9r_c, A9i_c = fourier_component(pos, mass, 9, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A10r, A10i, A10r_c, A10i_c = fourier_component(pos, mass, 10, Rmin, Rmax, 
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

    out['A0_c'] = A0_c
    out['A1r_c'], out['A1i_c'] = A1r_c, A1i_c
    out['A2r_c'], out['A2i_c'] = A2r_c, A2i_c
    out['A3r_c'], out['A3i_c'] = A3r_c, A3i_c
    out['A4r_c'], out['A4i_c'] = A4r_c, A4i_c
    out['A5r_c'], out['A5i_c'] = A5r_c, A5i_c
    out['A6r_c'], out['A6i_c'] = A6r_c, A6i_c
    out['A7r_c'], out['A7i_c'] = A7r_c, A7i_c
    out['A8r_c'], out['A8i_c'] = A8r_c, A8i_c
    out['A9r_c'], out['A9i_c'] = A9r_c, A9i_c
    out['A10r_c'], out['A10i_c'] = A10r_c, A10i_c

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
                    'A8r','A8i', 'A9r','A9i', 'A10r', 'A10i',
                    'A0_c', 'A1r_c', 'A1i_c',
                    'A2r_c','A2i_c', 'A3r_c','A3i_c',
                    'A4r_c','A4i_c', 'A5r_c','A5i_c',
                    'A6r_c','A6i_c', 'A7r_c','A7i_c',
                    'A8r_c','A8i_c', 'A9r_c','A9i_c', 'A10r_c', 'A10i_c']:
        #for key in ['Rlist', 'A0', 'A1r', 'A1i',
        #            'A2r','A2i', 'A3r','A3i',
        #            'A4r','A4i',
        #            'A0_c', 'A1r_c', 'A1i_c',
        #            'A2r_c','A2i_c', 'A3r_c','A3i_c',
        #            'A4r_c','A4i_c']:
            snap.create_dataset(key, data=t[key])
        time_list.append(t['time'])

    h5out.create_dataset('time', data=time_list)
    h5out.close()

    return None

def run(path, name, nsnap):
    fout = 'data/fourier_' + name + '.hdf5'

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
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'
    phgvS15Rc35 = 'phantom-vacuum-Sg15-Rc3.5'
    phgvS10Rc35 = 'phantom-vacuum-Sg10-Rc3.5'
    phgvS08Rc35 = 'phantom-vacuum-Sg08-Rc3.5'
    phgvS05Rc35 = 'phantom-vacuum-Sg05-Rc3.5'
    phgvS2Rc35star = 'phantom-vacuum-Sg20-Rc3.5-star'
    
    
    phgvS2Rc35RF = 'phantom-vacuum-Sg20-Rc3.5-RadFeed'
    phgS1 = 'phantom-Sg10-Rc4.0'

    NbodyBH = 'Nbody300-BH'
    NbodyBH1E8 = 'Nbody500-BH1E8'

    pair_list = [#(fid_dP, 'lvl5'), (fid_dP, 'lvl4'), #(fid_dP, 'lvl3'),
                 (Nbody, 'lvl4'), # 0
                 (Nbody, 'lvl3'), # 1
                 (Nbody, 'lvl2'), # 2
                 (phgvS2Rc35, 'lvl4'), # 3
                 (phgvS2Rc35, 'lvl3'), # 4
                 (phgvS2Rc35, 'lvl2'), # 5
                 (phgvS2Rc35, 'lvl3-snap700'), # 6
                 (phgvS2Rc35, 'lvl3-soft0.04'), # 7
                 #(phgvS2Rc35, 'lvl3-rstHalo'), (phgvS2Rc35, 'lvl3-snap700'),
                 #(phgvS2Rc35, 'lvl3-GFM'), (phgvS2Rc35, 'lvl3-isotherm'),
                 #(phgvS2Rc35, 'lvl4-GFM'),
                 (phgvS15Rc35, 'lvl3'), # 8
                 (phgvS10Rc35, 'lvl3'), # 9
                 (phgvS08Rc35, 'lvl3'), # 10
                 (phgvS05Rc35, 'lvl3'), # 11
                 (phgvS2Rc35star, 'lvl4'), # 12
                 (phgvS2Rc35star, 'lvl3'), # 13
                 (NbodyBH, 'lvl3'), # 14
                 (NbodyBH1E8, 'lvl3')] # 15
                 # (fid_dP_c1_bG, 'lvl5'), (fid_dP_c1_bG, 'lvl4'),# (fid_dP_c1_bG, 'lvl3'),
                 # (fid_dP_c1_bG1, 'lvl5'),# (fid_dP_c1_bG, 'lvl4'),# (fid_dP_c1_bG, 'lvl3'),
                 # (fid_dP_c1_rB, 'lvl5'), (fid_dP_c1_rB, 'lvl4'), (fid_dP_c1_rB, 'lvl3'),
                 # (fid_dP_c1_h, 'lvl5'), (fid_dP_c1_h, 'lvl4'), #(fid_dP_c1_h, 'lvl3'),
                 # (fid_dP_c1_h_v140, 'lvl5'), (fid_dP_c1_h_v140, 'lvl4')] #(fid_dP_c1_h, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                            
    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    i = int(sys.argv[2])
    path = path_list[i]
    name = name_list[i]
    nsnap = nsnap_list[i]

    run(path, name, nsnap)
