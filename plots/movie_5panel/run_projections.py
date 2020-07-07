from compute_projections import construct_update_projection_hdf5
import sys

try:
	nproc = int(sys.argv[1])
	print('running with nproc='+str(nproc))
except:
	print('running in serial mode')
	nproc=1

basepath = '../../runs/'

fid_dP_sEQ_g1 = 'fid-dispPoly-fg0.1-SoftEQS0'
fid_dP_rx_g1 = 'fid-dispPoly-fg0.1-MeshReg'
fid_dP_rx_g1_dAGB = 'fid-dispPoly-fg0.1-MeshReg-diskAGB'
fid_dP_rx_g1_dAGB_delay = 'fid-dispPoly-fg0.1-MeshReg-diskAGB-delay1.25'
fid_dP_rx_g1_Rcore = 'fid-dispPoly-fg0.1-MeshReg-Rcore1.0'
fid_dP_rx_g1_Rcore_AGB = 'fid-dispPoly-fg0.1-MeshReg-Rcore1.0-diskAGB'
fid_dP_rx_g1_Rcore14_AGB = 'fid-dispPoly-fg0.1-MeshReg-Rcore1.4-diskAGB'

pair_list = [(fid_dP_sEQ_g1, 'lvl4'), (fid_dP_sEQ_g1, 'lvl3'),
	  	     (fid_dP_rx_g1, 'lvl4'), (fid_dP_rx_g1, 'lvl3'),
             (fid_dP_rx_g1_Rcore, 'lvl4'), (fid_dP_rx_g1_Rcore, 'lvl3'),
             (fid_dP_rx_g1_Rcore_AGB, 'lvl4'), (fid_dP_rx_g1_Rcore_AGB, 'lvl3'),
             (fid_dP_rx_g1_Rcore14_AGB, 'lvl4')]#, (fid_dP_rx_g1_Rcore14_AGB, 'lvl3')]

name_list = [           p[0] + '-' + p[1] for p in pair_list]
path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                           
for name, path in zip(name_list, path_list):
    construct_update_projection_hdf5(name, path, nproc=nproc)
