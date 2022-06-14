from compute_projections import construct_update_projection_hdf5
import sys

try:
	nproc = int(sys.argv[1])
	print('running with nproc='+str(nproc))
except:
	print('running in serial mode')
	nproc=1

basepath = '../../hose_runs/'

fid_Rg15_V120 = 'fid-dispPoly-fg0.1-hose-Rg15.0-Rate1.0-Rh0.2-Vel120.0'
fid_Rg15_V160 = 'fid-dispPoly-fg0.1-hose-Rg15.0-Rate1.0-Rh0.2-Vel160.0'

pair_list = [(fid_Rg15_V120, 'lvl4'),
             (fid_Rg15_V160, 'lvl4')]

name_list = [           p[0] + '-' + p[1] for p in pair_list]
path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                           
for name, path in zip(name_list, path_list):
    construct_update_projection_hdf5(name, path, nproc=nproc, output_dir='data-hose/')
