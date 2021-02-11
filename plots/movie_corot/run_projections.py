from compute_projections import construct_update_projection_hdf5
import sys
import numpy as np

try:
	nproc = int(sys.argv[1])
	print('running with nproc='+str(nproc))
except:
	print('running in serial mode')
	nproc=1

basepath = '../../runs/'

Nbody = 'Nbody'

phgvS1 = 'phantom-vacuum-Sg10-Rc4.0'
phgvS2 = 'phantom-vacuum-Sg20-Rc4.0'

pair_list = [(Nbody, 'lvl3'),
             (phgvS1, 'lvl3'),
             (phgvS2, 'lvl3')]

name_list = [           p[0] + '-' + p[1] for p in pair_list]
path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                           
fourier_path = '../fourier_component/data/'

for name, path in zip(name_list, path_list):
    if 'Nbody' in name:
        center = np.array([0, 0, 0])
    else:
        center = np.array([200, 200, 200])

    if 'phantom' in name:
        firstkey = 0
    else:
        firstkey = 150

    construct_update_projection_hdf5(name, path, nproc=nproc, center=center, 
                                     corot=True, fourier_path=fourier_path, firstkey=firstkey)
