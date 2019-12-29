import numpy as np
import arepo
import sys
from tqdm import tqdm

def compute_veldisp(path, snapnum, name, output_dir='data/'):
    # try loading snapshot
    try:
        sn = arepo.Snapshot(path+'output/', snapnum, combineFiles=True)
    except:
        print("unable to load name:", name, "path:"+path, " snapnum: ", snapnum)
        return None
    
    return sn


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
            out = compute_veldisp(path, snapnum, name)
    