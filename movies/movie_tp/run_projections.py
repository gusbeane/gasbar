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
Nbody_tp = 'Nbody-tp'

pair_list = [(Nbody, 'lvl4'),
             (Nbody_tp, 'lvl4-1x'), (Nbody_tp, 'lvl4-8x'), (Nbody_tp, 'lvl4-64x')]

name_list = [           p[0] + '-' + p[1] for p in pair_list]
path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                           
for name, path in zip(name_list, path_list):
    if 'Nbody' in name:
        center = np.array([0, 0, 0])
    else:
        center = np.array([200, 200, 200])

    parttype = [2, 3, 5]

    construct_update_projection_hdf5(name, path, nproc=nproc, center=center, parttype=parttype)
