import numpy as np
import arepo
import astropy.units as u

import sys
from tqdm import tqdm

def compute_com(path, snapnum, name, output_dir='data/'):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'output/', snapnum, combineFiles=True)
    except:
        print("unable to load name:", name, "path:"+path, " snapnum: ", snapnum)
        return None

    firstpart = True

    # loop through all particle types
    for i, npart in enumerate(sn.NumPart_Total):
        if npart == 0:
            continue

        part = getattr(sn, 'part'+str(i))

        # compute the center of mass
        mass = sn.MassTable[i].as_unit(arepo.u.msol).value * u.Msun
        pos = part.pos.as_unit(arepo.u.kpc).value * u.kpc

        # if mass is zero, then we need to load each individual mass
        if mass == 0 * u.Msun:
            mass = part.mass.as_unit(arepo.u.msol).value * u.Msun
            this_com = np.average(pos, axis=0, weights=mass)
            this_mass = np.sum(mass)
        else:
            this_com = np.average(pos, axis=0)
            this_mass = mass * npart

        if firstpart:
            com_mass = this_mass.copy()
            com_by_mass = this_com.copy() * this_mass

            firstpart = False
        else:
            com_by_mass += this_com * this_mass
            com_mass += this_mass

        return com_by_mass/com_mass



if __name__ == '__main__':

    basepath = '../../runs/'

    nbody = 'fid-Nbody/'
    wet = 'fid-wet/'
    fid = 'fid/'
    
    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        path_list = [basepath + nbody + 'lvl5/']
        name_list = ['nbody-lvl5']
    else:
        lvl_list = [5, 4, 3, 2]
        path_list = [basepath + nbody + 'lvl' + str(i) + '/' for i in lvl_list]
        name_list = ['nbody-lvl' + str(i) for i in lvl_list]
    
    snapnum_list = [0, 100, 200, 300, 400, 500, 600]

    for path, name in zip(tqdm(path_list), name_list):
        for snapnum in snapnum_list:
            out = compute_com(path, snapnum, name)
            print(name, 'snap:', snapnum, out)
