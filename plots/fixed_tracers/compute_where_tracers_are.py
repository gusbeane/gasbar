import numpy as np
import arepo
import sys
from tqdm import tqdm
import pickle

def return_initial_and_final_pos(path, name, rcut, snapnum1, snapnum2, 
                                 output_dir='data/', tracer_type=5):
    # try loading snapshot
    sn1 = arepo.Snapshot(path+'output/', snapnum1, combineFiles=True)
    sn2 = arepo.Snapshot(path+'output/', snapnum2, combineFiles=True)
    rcut_string = "{:.2f}".format(rcut)
    
    # now load in all the tracers from the second snapshot
    


if __name__ == '__main__':

    basepath = '../../runs/'

    fid_fixed = ''
    
    pair_list = (fid_fixed, 'lvl5')

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
        
    snapnum1 = 308
    snapnum2 = 358
    rcut=3.0

    for path, name in zip(tqdm(path_list), name_list):
        save_initial_and_final_pos(path, name, rcut, snapnum1, snapnum2)
    