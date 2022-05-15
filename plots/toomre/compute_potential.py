import numpy as np
import arepo
import sys
from tqdm import tqdm
import agama

agama.setUnits(mass=1, length=1, velocity=1)

# HASNT BEEN USED AT ALL

def construct_potential_of_type(snap, ptype_list, center=np.array([0, 0, 0]), fout_base=None):
    if isinstance(ptype_list, str):
        ptype_list = (ptype_list,)

    pot_list = []
    for ptype in ptype_list:
        part = getattr(snap, 'PartType'+str(ptype))
        if len(part.data) == 0: # contains no particles
            continue
        pos = part['Coordinates']
        if 

        if ptype == 'PartType1' or ptype == 'PartType3':
            pot = agama.Potential(type='Multipole', particles=(pos, mass),
                                  symmetry='a', gridsizeR=20, lmax=2)
        
        else:
            pot = agama.Potential(type='CylSpline', particles=(pos, mass),
                                  symmetry='a', gridsizer=20, gridsizez=20, mmax=0, 
                                  Rmin=0.2, Rmax=50, Zmin=0.02, Zmax=10)
        
        pot_list.append(pot)

    return pot_list

def compute_potential(path, snapnum, name, center=np.array([0, 0, 0]), output_dir='data/'):
    # try loading snapshot
    sn = arepo.Snapshot(path+'output/', snapnum, combineFiles=True)


    return sn


if __name__ == '__main__':

    basepath = '../../runs/'

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_d15_g1 = 'fid-disp1.5-fg0.1'

    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        pair_list = [(fid_g1, 'lvl5')]
    else:
        pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), (fid_g1, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                            
    # nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]
    snap_list = np.arange(0, 100, 10)

    
    snapnum_list = [10, 50, 100, 150, 200, 300, 400, 500, 600]

    for path, name in zip(tqdm(path_list), name_list):
        for snapnum in snap_list:
            out = compute_potential(path, snapnum, name, center=np.array([200, 200, 200]), output_dir='pot/')
    