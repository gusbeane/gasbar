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
def gas_flow(pos, mass, vel, Rmin, Rmax, nbins=20, logspace=True):
    if logspace:
        bins = np.linspace(np.log10(Rmin), np.log10(Rmax), nbins+1)
        bins = np.power(10., bins)
    else:
        bins = np.linspace(Rmin, Rmax, nbins+1)
    
    flow = np.zeros(nbins)
    Rmag = np.zeros(nbins)
    N_in_bin = np.zeros(nbins)
    
    Npart = len(pos)
    for i in range(Npart):
        R = np.sqrt(pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1])
        cphi = pos[i][0]/R
        sphi = pos[i][1]/R

        vR = vel[i][0] * cphi + vel[i][1] * sphi
        
        for j in range(nbins):
            if R > bins[j] and R < bins[j+1]:
                flow[j] += mass[j] * vR
                Rmag[j] += R
                N_in_bin[j] += 1
    
    
    for j in range(nbins):
        if N_in_bin[j] > 0:
            Rmag[j] /= N_in_bin[j]
        else:
            Rmag[j] = np.nan 
            flow[j] = np.nan
    
    return Rmag, flow

def compute_gas_flow(path, snapnum, Rmin=0.0, Rmax=30.0, nbins=60, logspace=False, center=None):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True, 
                            parttype=[0], fields=['Coordinates', 'Masses', 'Velocities'])
    except:
        print("unable to load path:"+path, " snapnum: ", snapnum)
        return None
    
    firstpart = True
    
    pos = sn.part0.pos
    if center is not None:
        pos = np.subtract(pos, center)
    vel = sn.part0.vel
    mass = sn.part0.mass

    Rlist, flow_rate = gas_flow(pos, mass, vel, Rmin, Rmax, 
                                       nbins=nbins, logspace=logspace)
    
    time = sn.Time.as_unit(arepo.u.d).value * u.d
    time = time.to_value(u.Myr)

    out = {}
    out['Rlist'] = Rlist
    out['flow_rate'] = flow_rate
    out['time'] = time

    return out

def concat_files(outs, indices, fout):
    h5out = h5.File(fout, mode='w')
    time_list = []

    for t, idx in zip(outs, indices):
        snap = h5out.create_group('snapshot_'+"{:03d}".format(idx))

        for key in ['Rlist', 'flow_rate']:
            snap.create_dataset(key, data=t[key])
        time_list.append(t['time'])

    h5out.create_dataset('time', data=time_list)
    h5out.close()

    return None

def run(path, name, nsnap):
    fout = 'data/gasflow_' + name + '.hdf5'

    # dont remake something already made
    if os.path.exists(fout):
        return None

    if 'Nbody' in name:
        center = None
    else:
        center = np.array([200, 200, 200])

    indices = np.arange(nsnap)
    outs = Parallel(n_jobs=nproc) (delayed(compute_gas_flow)(path, int(idx), center=center) for idx in tqdm(indices))

    concat_files(outs, indices, fout)
    

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    fid_dP = 'fRpoly'
    fid_dP_c1 = 'fRpoly-Rcore1.0'
    fid_dP2_c1 = 'fRpoly2-Rcore1.0'
    fid_dP_c1_rx = 'fRpoly-Rcore1.0-relax'
    fid_dP_c1_bG = 'fRpoly-Rcore1.0-barGas'
    fid_dP_c1_bG1 = 'fRpoly-Rcore1.0-barGas1.0'
    fid_dP_c1_rB = 'fRpoly-Rcore1.0-ringBug'
    fid_dP_c1_h = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel160.0'
    fid_dP_c1_h_v140 = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel140.0'


    pair_list = [#(fid_dP, 'lvl5'), (fid_dP, 'lvl4'), #(fid_dP, 'lvl3'),
                 (fid_dP_c1, 'lvl5'), (fid_dP_c1, 'lvl4'), (fid_dP_c1, 'lvl3'),
                 (fid_dP2_c1, 'lvl5'), (fid_dP2_c1, 'lvl4'), (fid_dP2_c1, 'lvl3')]
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
