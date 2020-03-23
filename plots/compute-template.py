import numpy as np
import arepo
import sys
from tqdm import tqdm

def compute_REPLACE(path, snapnum, name, output_dir='data/'):
    # try loading snapshot
    sn = arepo.Snapshot(path+'output/', snapnum, combineFiles=True)

    return sn


if __name__ == '__main__':

    basepath = '../../runs/'

    fid_g1 = 'fid-disp1.0-fg0.1'

    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        pair_list = [(fid_g1, 'lvl5')]
    else:
        pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), (fid_g1, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                            
    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    # if len(sys.argv) == 3:
    #     i = int(sys.argv[2])
    #     path = path_list[i]
    #     name = name_list[i]
    #     nsnap = nsnap_list[i]

    #     run(path, name, nsnap)
    # else:
    #     for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
    #         run(path, name, nsnap)


    
    snapnum_list = [10, 50, 100, 150, 200, 300, 400, 500, 600]

    for path, name in zip(tqdm(path_list), name_list):
        for snapnum in snapnum_list:
            out = compute_REPLACE(path, snapnum, name)
    