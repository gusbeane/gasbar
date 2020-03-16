import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import h5py as h5
import glob
import os

from joblib import Parallel, delayed

def fourier_component(pos, mass, m, Rmin, Rmax, nbins=20, logspace=True, amplitude=True):

    assert isinstance(m, int), "m must be an integer!"
    assert isinstance(float(Rmin), float), "Rmin must be a float!"
    assert isinstance(float(Rmax), float), "Rmax must be a float!"

    if logspace:
        bins = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins)
    else:
        bins = np.linspace(Rmin, Rmax, nbins)

    # loop through types and concatenate positions and masses    
    Rmag = np.linalg.norm(pos[:,:2], axis=1)

    keys = np.logical_and(Rmag > Rmin, Rmag < Rmax)

    # get output R list by averaging the value of R in each bin
    digit = np.digitize(Rmag[keys], bins)
    R_list = [Rmag[keys][digit == i].mean() for i in range(1, len(bins))]

    # compute the Am term for each particle
    phi_list = np.arctan2(pos[:,1], pos[:,0])
    Am_i = mass*np.exp((1j)*m*phi_list)

    if amplitude:
        Am_list = [np.abs(np.sum(Am_i[keys][digit == i])) for i in range(1, len(bins))]
        return np.array(R_list), np.array(Am_list)
    else:
        Am_list_real = [np.sum(np.real(Am_i[keys][digit == i])) for i in range(1, len(bins))]
        Am_list_imag = [np.sum(np.imag(Am_i[keys][digit == i])) for i in range(1, len(bins))]
        return np.array(R_list), np.array(Am_list_real), np.array(Am_list_imag)

def compute_fourier_component(path, snapnum, Rmin=0.0, Rmax=30.0, nbins=60, logspace=False, center=None):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True)
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

    Rlist, A0 = fourier_component(pos, mass, 0, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    Rlist, A1r, A1i = fourier_component(pos, mass, 1, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A2r, A2i = fourier_component(pos, mass, 2, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A3r, A3i = fourier_component(pos, mass, 3, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A4r, A4i = fourier_component(pos, mass, 4, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A5r, A5i = fourier_component(pos, mass, 5, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A6r, A6i = fourier_component(pos, mass, 6, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A7r, A7i = fourier_component(pos, mass, 7, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A8r, A8i = fourier_component(pos, mass, 8, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A9r, A9i = fourier_component(pos, mass, 9, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    Rlist, A10r, A10i = fourier_component(pos, mass, 10, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace, amplitude=False)
    
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

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_g2 = 'fid-disp1.0-fg0.2'
    fid_g3 = 'fid-disp1.0-fg0.3'
    fid_g4 = 'fid-disp1.0-fg0.4'
    fid_g5 = 'fid-disp1.0-fg0.5'    
    fid_d7_g3 = 'fid-disp0.7-fg0.3'
    fid_d5_g3 = 'fid-disp0.5-fg0.3'
    fid_g3_nB = 'fid-disp1.0-fg0.3-noBulge' 
    fid_g1_da = 'fid-disp1.0-fg0.1-diskAcc1.0'
    fid_g1_da_am = 'fid-disp1.0-fg0.1-diskAcc1.0-decAngMom' 
    fid_g1_corona = 'fid-disp1.0-fg0.1-corona'
    fid_g1_coronaRot = 'fid-disp1.0-fg0.1-coronaRot'
    fid_g1_coronaMat = 'fid-disp1.0-fg0.1-corona-Matthew'
    fid_g1_coronaMat4 = 'fid-disp1.0-fg0.1-corona-Matthew-MHG0.004'
    
    fid_g1_fixed1kpc = 'fid-disp1.0-fixedDisk-core1kpc'
    fid_g1_fixed2kpc = 'fid-disp1.0-fixedDisk-core2kpc'
    fid_g1_fixed3kpc = 'fid-disp1.0-fixedDisk-core3kpc'
    fid_g1_fixed4kpc = 'fid-disp1.0-fixedDisk-core4kpc' 
    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        pair_list = [(fid_g1, 'lvl5')]
    else:
        pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), (fid_g1, 'lvl3'),
                     #(fid_g2, 'lvl5'), (fid_g2, 'lvl4'), (fid_g2, 'lvl3'),
                     #(fid_g3, 'lvl5'), (fid_g3, 'lvl4'), (fid_g3, 'lvl3'),
                     #(fid_g4, 'lvl5'), (fid_g4, 'lvl4'),
                     #(fid_g5, 'lvl5'), (fid_g5, 'lvl4'),
                     (fid_g1_fixed1kpc, 'lvl5'), (fid_g1_fixed1kpc, 'lvl4'),
                     (fid_g1_fixed2kpc, 'lvl5'), (fid_g1_fixed2kpc, 'lvl4'),
                     (fid_g1_fixed3kpc, 'lvl5'), (fid_g1_fixed3kpc, 'lvl4'),
                     (fid_g1_fixed4kpc, 'lvl5'), (fid_g1_fixed4kpc, 'lvl4')]
                             
                     #(fid_g1_corona, 'lvl5'), (fid_g1_corona, 'lvl4'),
                     #(fid_g1_coronaRot, 'lvl5'), (fid_g1_coronaRot, 'lvl4'),
                     #(fid_g1_coronaMat, 'lvl5'), (fid_g1_coronaMat, 'lvl4'),
                     #(fid_g1_coronaMat4, 'lvl5'), (fid_g1_coronaMat4, 'lvl4')]
                     #(fid_d7_g3, 'lvl5'), (fid_d7_g3, 'lvl4'),
                     #(fid_d5_g3, 'lvl5'), (fid_d5_g3, 'lvl4'),
                     #(fid_g3_nB, 'lvl5'), (fid_g3_nB, 'lvl4'),
                     #(fid_g1_da, 'lvl5'), (fid_g1_da, 'lvl4'),
                     #(fid_g1_da_am, 'lvl5'), (fid_g1_da_am, 'lvl4')]
    
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
