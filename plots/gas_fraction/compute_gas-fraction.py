import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import pickle
from joblib import Parallel, delayed

def compute_gas_fraction(path, name, skip=50, output_dir='data/',
                         Rmax=20, zmax=3, 
                         R0=8.2, dR=1.0, Rcenter=6.0,
                         gas_ptype=0, star_ptype=(2,3,4),
                         center = np.array([200, 200, 200])):
    # try loading snapshot
    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))

    out = []
    for snapnum in tqdm(np.arange(nsnap)[::skip]):
        sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True)
        time = sn.header.time.as_unit(3.1536E13 * arepo.u.second).value # in Myr

        gas_mass = get_mass_in_ptype(sn, gas_ptype, 0., Rmax, 0., zmax, center)
        star_mass = get_mass_in_ptype(sn, star_ptype, 0., Rmax, 0., zmax, center)

        gas_fraction_disk = gas_mass/(gas_mass + star_mass)

        gas_mass = get_mass_in_ptype(sn, gas_ptype, R0-dR/2., R0+dR/2., 0., zmax, center)
        star_mass = get_mass_in_ptype(sn, star_ptype, R0-dR/2., R0+dR/2., 0., zmax, center)

        gas_fraction_R0 = gas_mass/(gas_mass + star_mass)
        
        gas_mass = get_mass_in_ptype(sn, gas_ptype, 0., Rcenter, 0., zmax, center)
        star_mass = get_mass_in_ptype(sn, star_ptype, 0., Rcenter, 0., zmax, center)

        gas_fraction_Rcenter = gas_mass/(gas_mass + star_mass)

        out.append([time, gas_fraction_disk, gas_fraction_R0, gas_fraction_Rcenter])

    pickle.dump(np.array(out), open(output_dir+'gas-fraction_'+name+'.p', 'wb'))

def get_mass_in_ptype(snap, ptype, Rmin, Rmax, zmin, zmax, center):
    if isinstance(ptype, int):
        ptype = (ptype,)

    totmass = 0.0
    for pt in ptype:

        if snap.NumPart_Total[pt] ==0:
            continue

        part = getattr(snap, 'part'+str(pt))

        part_pos = part.pos.as_unit(arepo.u.kpc).value
        part_pos = np.subtract(part.pos, center)
    
        R = np.sqrt( np.add( np.square(part_pos[:,0]), np.square(part_pos[:,1]) ) )
        z = np.abs(part_pos[:,2])
    
        Rbool = np.less_equal(R, Rmax)
        zbool = np.less_equal(z, zmax)
        keys = np.logical_and(Rbool, zbool)

        if hasattr(part, 'mass'):
            mass_in_pt = np.sum(part.mass[keys]).as_unit(arepo.u.msol).value
        else:
            mass_in_pt = len(np.where(keys)[0]) * snap.MassTable[pt].as_unit(arepo.u.msol).value

        totmass += mass_in_pt

    return totmass


if __name__ == '__main__':

    basepath = '../../runs/'

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_g2 = 'fid-disp1.0-fg0.2'
    fid_g3 = 'fid-disp1.0-fg0.3'
    fid_g4 = 'fid-disp1.0-fg0.4'
    fid_g5 = 'fid-disp1.0-fg0.5'

    fid_da = 'fid-disp1.0-fg0.1-diskAcc1.0'
    fid_da_am = 'fid-disp1.0-fg0.1-diskAcc1.0-decAngMom'

    fid_g1_dS_out_delay = 'fid-disp1.0-fg0.1-diskAGB-outer-delay1.0'

    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        nproc=2
        pair_list = [(fid_g1, 'lvl5'), (fid_g2, 'lvl5')]
    else:
        nproc=16
        pair_list = [#(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), #(fid_g1, 'lvl3'),
                     #(fid_g2, 'lvl5'), (fid_g2, 'lvl4'), (fid_g2, 'lvl3'),
                     #(fid_g3, 'lvl5'), (fid_g3, 'lvl4'), (fid_g3, 'lvl3'),
                     #(fid_g4, 'lvl5'), (fid_g4, 'lvl4'),
                     #(fid_g5, 'lvl5'), (fid_g5, 'lvl4'),
                     #(fid_da, 'lvl5'), (fid_da, 'lvl4'),
                     #(fid_da_am, 'lvl5'), (fid_da_am, 'lvl4'),
                     (fid_g1_dS_out_delay, 'lvl5'), (fid_g1_dS_out_delay, 'lvl4')]
    
    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
    
    Parallel(n_jobs=nproc) (delayed(compute_gas_fraction)(path, name) 
                            for path,name in zip(path_list, name_list))
