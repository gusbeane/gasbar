import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import h5py as h5

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

def compute_fourier_component(path, snapnum, Rmin=0.0, Rmax=30.0, nbins=60, logspace=False):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'output/', snapnum, combineFiles=True)
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

if __name__ == '__main__':

    basepath = '../../runs/'

    nbody = 'fid-Nbody/'
    wet = 'fid-wet/'
    fid = 'fid/'

    nproc = int(sys.argv[1])
    
    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        path_list = [basepath + nbody + 'lvl5/']
        name_list = ['nbody-lvl5']
        final_frame_list = [20]
    else:
        if sys.argv[1] == 'lvl2':
            lvl_list = [2]
        else:
            lvl_list = [5, 4, 3]
        path_list = [basepath + nbody + 'lvl' + str(i) + '/' for i in lvl_list]
        name_list = ['nbody-lvl' + str(i) for i in lvl_list]
        final_frame_list = [620, 620, 620, 122]
    
    for path, name, final_frame in zip(tqdm(path_list), name_list, final_frame_list):
        indices = np.arange(final_frame+1)
        outs = Parallel(n_jobs=nproc) (delayed(compute_fourier_component)(path, int(idx)) for idx in tqdm(indices))

        concat_files(outs, indices, 'data/fourier_' + name + '.hdf5')
