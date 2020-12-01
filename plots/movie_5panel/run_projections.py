from compute_projections import construct_update_projection_hdf5
import sys

try:
	nproc = int(sys.argv[1])
	print('running with nproc='+str(nproc))
except:
	print('running in serial mode')
	nproc=1

basepath = '../../runs/'

fid_dP = 'fRpoly'
fid_dP_c1 = 'fRpoly-Rcore1.0'
fid_dP_c1_h = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel160.0'
fid_dP_c1_h_v140 = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel140.0'


pair_list = [(fid_dP, 'lvl5'), (fid_dP, 'lvl4'), #(fid_dP, 'lvl3'),
             (fid_dP_c1, 'lvl5'), (fid_dP_c1, 'lvl4'), #(fid_dP_c1, 'lvl3'),
             (fid_dP_c1_h, 'lvl5'), (fid_dP_c1_h, 'lvl4'), #(fid_dP_c1_h, 'lvl3'),
             (fid_dP_c1_h_v140, 'lvl5'), (fid_dP_c1_h_v140, 'lvl4')] #(fid_dP_c1_h, 'lvl3')]

name_list = [           p[0] + '-' + p[1] for p in pair_list]
path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                           
for name, path in zip(name_list, path_list):
    construct_update_projection_hdf5(name, path, nproc=nproc)
