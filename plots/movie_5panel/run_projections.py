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
fid_dP = 'fRpoly'
fid_dP_c1 = 'fRpoly-Rcore1.0'
fid_dP2_c1 = 'fRpoly2-Rcore1.0'
fid_dP_c1_rx = 'fRpoly-Rcore1.0-relax'
#fid_dP_c1_bG = 'fRpoly-Rcore1.0-barGas'
#fid_dP_c1_bG1 = 'fRpoly-Rcore1.0-barGas1.0'
fid_dP_c1_bG2 = 'fRpoly-Rcore1.0-barGas2.0'
fid_dP_c1_sp = 'fRpoly-Rcore1.0-spring'
fid_dP_c1_MB = 'fRpoly-Rcore1.0-MB0.004'

fid_dP_c1_MB_sp = 'fRpoly-Rcore1.0-MB0.004-spring'
fid_dP_c1_MB_eff2 = 'fRpoly-Rcore1.0-MB0.004-FeedEff2.0'
fid_dP_c1_MB_LM = 'fRpoly-Rcore1.0-MB0.004-LumMass10000'
fid_dP_c1_noMB = 'fRpoly-Rcore1.0-noMB'

fid_dP_c1_MB_WSF2 = 'fRpoly-Rcore1.0-MB0.004-WindSpeedFactor2'
fid_dP_c1_MB_LM3 = 'fRpoly-Rcore1.0-MB0.004-LumMass3000'
fid_dP_noB_fR12 = 'fR1.2-fg0.1-noMB-MD0.05'
fid_dP_MB2_fR12 = 'fR1.2-fg0.1-MB0.002-MD0.05-JD0.0521'

f_g1_MB = 'fR1.3-fg0.1-MB0.004'
f_g2_MB = 'fR1.4-fg0.2-MB0.004'
fR3_g2_MB = 'fR3.0-fg0.2-MB0.004'

f_Ngb32  = 'fRpoly-Rcore1.0-DesNumNgbEnrichment32'
f_Ngb128 = 'fRpoly-Rcore1.0-DesNumNgbEnrichment128'
f_Ngb512 = 'fRpoly-Rcore1.0-DesNumNgbEnrichment512'
f_Ngb512_ft = 'fRpoly-Rcore1.0-DesNumNgbEnrichment512-ftrap'
f_Ngb512_rf = 'fRpoly-Rcore1.0-DesNumNgbEnrichment512-RadFeed'

f_noMB_Ngb512 = 'fRpoly-Rcore1.0-noMB-Ngb512'

f_Sg40 = 'fR1.0-Sg40-Rc4.5'
f_Sg10 = 'fR1.0-Sg10-Rc4.5'

fid_dP_c1_rB = 'fRpoly-Rcore1.0-ringBug'
fid_dP_c1_h = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel160.0'
fid_dP_c1_h_v140 = 'fRpoly-Rcore1.0-hose-Del1.0-Rg15.0-Rate0.5-Rh0.2-Vel140.0'

ph = 'phantom'
phNgb = 'phantom-Ngb512'
phg = 'phantom-grav'
phv = 'phantom-vacuum'
phgv = 'phantom-vacuum-grav'

phgvS1 = 'phantom-vacuum-Sg10-Rc4.0'
phgvS2 = 'phantom-vacuum-Sg20-Rc4.0'
phgS1 = 'phantom-Sg10-Rc4.0'
phgvS1Ngbft = 'phantom-vacuum-Sg10-Rc4.0-DesNumNgbEnrichment512-ftrap'


pair_list = [(Nbody, 'lvl5'), (Nbody, 'lvl4'), (Nbody, 'lvl3'),
             (fid_dP, 'lvl5'), (fid_dP, 'lvl4'), #(fid_dP, 'lvl3'),
             (fid_dP_c1, 'lvl5'), (fid_dP_c1, 'lvl4'), (fid_dP_c1, 'lvl3'),
             (fid_dP2_c1, 'lvl5'), (fid_dP2_c1, 'lvl4'), (fid_dP2_c1, 'lvl3'),
             #(fid_dP_c1_rx, 'lvl5'), (fid_dP_c1_rx, 'lvl4'),# (fid_dP_c1_rx, 'lvl3'),
             #(fid_dP_c1_rx, 'lvl4-snap001'),# (fid_dP_c1_rx, 'lvl3'),
             #(fid_dP_c1_bG, 'lvl5'), (fid_dP_c1_bG, 'lvl4'),# (fid_dP_c1_bG, 'lvl3'),
             #(fid_dP_c1_bG1, 'lvl5'),# (fid_dP_c1_bG, 'lvl4'),# (fid_dP_c1_bG, 'lvl3'),
             (fid_dP_c1_bG2, 'lvl5'), (fid_dP_c1_bG2, 'lvl4'), (fid_dP_c1_bG2, 'lvl3'),
             (fid_dP_c1_sp, 'lvl5'), (fid_dP_c1_sp, 'lvl4'), (fid_dP_c1_sp, 'lvl3'),
             (fid_dP_c1_MB, 'lvl5'), (fid_dP_c1_MB, 'lvl4'), (fid_dP_c1_MB, 'lvl3'),
             (fid_dP_c1_MB_sp, 'lvl5'), (fid_dP_c1_MB_sp, 'lvl4'), (fid_dP_c1_MB_sp, 'lvl3'),
             (fid_dP_c1_MB_eff2, 'lvl5'), (fid_dP_c1_MB_eff2, 'lvl4'), (fid_dP_c1_MB_eff2, 'lvl3'),
             (fid_dP_c1_MB_LM, 'lvl4'), (fid_dP_c1_MB_LM, 'lvl3'),
             (fid_dP_c1_noMB, 'lvl5'), (fid_dP_c1_noMB, 'lvl4'), (fid_dP_c1_noMB, 'lvl3'), (fid_dP_c1_noMB, 'lvl3-TauBug'),
             (fid_dP_c1_MB_WSF2, 'lvl4'), (fid_dP_c1_MB_WSF2, 'lvl3'),
             (fid_dP_c1_MB_LM3, 'lvl4'), (fid_dP_c1_MB_LM3, 'lvl3'),
             (fid_dP_noB_fR12, 'lvl4'), (fid_dP_noB_fR12, 'lvl3'),
             (fid_dP_MB2_fR12, 'lvl4'), (fid_dP_MB2_fR12, 'lvl3'),
             (f_g1_MB, 'lvl4'), (f_g1_MB, 'lvl3'),
             (f_g2_MB, 'lvl4'), (f_g2_MB, 'lvl3'),
             (fR3_g2_MB, 'lvl4'), (fR3_g2_MB, 'lvl3'),
             (f_Ngb32, 'lvl4'), (f_Ngb32, 'lvl3'),
             (f_Ngb128, 'lvl4'), (f_Ngb128, 'lvl3'),
             (f_Ngb512, 'lvl4'), (f_Ngb512, 'lvl3'),
             (f_Ngb512_ft, 'lvl3-ft0'), (f_Ngb512_ft, 'lvl3-ft2'), (f_Ngb512_ft, 'lvl3-ft8'),
             (f_Ngb512_rf, 'lvl3'),
             (f_noMB_Ngb512, 'lvl3'),
             (f_Sg40, 'lvl4'), (f_Sg40, 'lvl3'),
             (f_Sg10, 'lvl4'), (f_Sg10, 'lvl3'),
             (ph, 'lvl3'),
             (phNgb, 'lvl3'),
             (phv, 'lvl3'),# (phv, 'lvl3-grav'),
             (phg, 'lvl3'),
             (phgv, 'lvl3'),
             (phgvS1, 'lvl3'),
             (phgvS2, 'lvl3'),
             (phgS1, 'lvl3'),
             (phgvS1Ngbft, 'lvl3-ft8')]
             #(fid_dP_c1_rB, 'lvl5'), (fid_dP_c1_rB, 'lvl4'), (fid_dP_c1_rB, 'lvl3'),
             #(fid_dP_c1_h, 'lvl5'), (fid_dP_c1_h, 'lvl4'), #(fid_dP_c1_h, 'lvl3'),
             #(fid_dP_c1_h_v140, 'lvl5'), (fid_dP_c1_h_v140, 'lvl4')] #(fid_dP_c1_h, 'lvl3')]

name_list = [           p[0] + '-' + p[1] for p in pair_list]
path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                           
for name, path in zip(name_list, path_list):
    if 'Nbody' in name:
        center = np.array([0, 0, 0])
    else:
        center = np.array([200, 200, 200])
    construct_update_projection_hdf5(name, path, nproc=nproc, center=center)
