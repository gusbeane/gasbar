import os
import glob

import numpy as np 
import h5py as h5
from joblib import Parallel, delayed

import arepo
from tqdm import tqdm
from numba import njit
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter

@njit
def compute_surface_density(R, mass, Rbins):
    key = np.where(R < Rbins[-1])[0]
    R = R[key]
    
    surf_dens = np.zeros(len(Rbins)-1)
    ave_R = np.zeros(len(Rbins)-1)
    N_in_bin = np.zeros(len(Rbins)-1)
    
    for i in range(len(R)):
        for j in range(len(Rbins)-1):
            if R[i] >= Rbins[j] and R[i] < Rbins[j+1]:
                ave_R[j] += R[i]
                N_in_bin[j] += 1
                surf_dens[j] += mass
    
    for j in range(len(Rbins)-1):
        if N_in_bin[j] > 0:
            ave_R[j] /= N_in_bin[j]
            surf_dens[j] /= np.pi * (Rbins[j+1]**2 - Rbins[j]**2)
        else:
            ave_R[j] = np.nan
            surf_dens[j] = np.nan
    
    return ave_R, surf_dens

def make_projection_snap(path, snapnum, parttype=[1], 
                         center=np.array([200, 200, 200]), width=30., nres=256):

    sn = arepo.Snapshot(path+'/output', snapnum, parttype=parttype, 
                        combineFiles=True, fields=['Coordinates', 'Masses'])

    time = sn.Time.value

    range_xy = [[center[0] - width/2.0, center[0] + width/2.0], [center[1] - width/2.0, center[1] + width/2.0]]

    surf = (width/nres)**(2.0)

    heatmap_xy_out = []
    for pt in parttype:
        if sn.NumPart_Total[pt] == 0:
            heatmap_xy_out.append(np.zeros((nres, nres)))
            continue

        part = getattr(sn, 'part'+str(pt))

        pos = part.pos.value - center

        if sn.MassTable[pt] > 0:
            weights = None
            postfac = sn.MassTable[pt] / surf
        else:
            weights = part.Masses / surf
            postfac = 1.0

        heatmap_xy, xbins, ybins = np.histogram2d(pos[:,0], pos[:,1], bins=(nres, nres), range=range_xy, weights=weights)
        heatmap_xy *= postfac

        Rbins = np.logspace(-3, 2, 80)
        R = np.linalg.norm(pos[:,:2], axis=1)
        ave_R, surf = compute_surface_density(R, sn.MassTable[pt], Rbins)

        surf_interp = interp1d(ave_R, surf, fill_value='extrapolate')

        ave_x = (xbins[:-1] + xbins[1:])/2.0
        ave_y = (ybins[:-1] + ybins[1:])/2.0
        xgrid, ygrid = np.meshgrid(ave_x, ave_y, indexing='ij')
        Rgrid = np.sqrt(xgrid * xgrid + ygrid * ygrid)
        surf_grid = surf_interp(Rgrid)

        heatmap_xy -= surf_grid
        heatmap_xy = gaussian_filter(heatmap_xy, sigma=8)

        heatmap_xy_out.append(heatmap_xy)

    return heatmap_xy_out, time

def do_i_continue(fname, nsnap, parttype):
    if not os.path.exists(fname):
        return True
    
    f = h5.File(fname, mode='r')
    for pt in parttype:
        pt_str = str(pt)
        for snap in range(nsnap):
            snap_key = 'snapshot_'+"{:03d}".format(snap)
            if snap_key not in f['PartType' + pt_str + '/xy'].keys():
                f.close()
                return True

    # all projections already computed
    f.close()
    return False

def construct_update_projection_hdf5(name, path, nproc=1, parttype=[1], center=np.array([200., 200., 200.]),
                                     width=30., nres=256, output_dir='data/'):

    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    assert nsnap > 0,"No output files detected for name="+name

    fname = name + '_w' + "{:.01f}".format(width) + '_n' + str(nres) + '.hdf5' 

    if not do_i_continue(output_dir + '/' + fname, nsnap, parttype):
        return
    
    f = h5.File(output_dir + '/' + fname, mode='a')

    if 'width' not in f.attrs.keys():
        f.attrs['width'] = width
    if 'nres' not in f.attrs.keys():
        f.attrs['nres'] = nres

    for pt in parttype:
        if 'PartType' + str(pt) not in f.keys():
            f.create_group('PartType' + str(pt)+'/xy')

    snap_list = []
    for snap in range(nsnap):
        snap_key = 'snapshot_'+"{:03d}".format(snap)
        if snap_key not in f['PartType'+str(parttype[0])+'/xy']:
            snap_list.append(snap)

    out = Parallel(n_jobs=nproc) (delayed(make_projection_snap)(path, snap, parttype=parttype,
                                                                center=center, width=width, nres=nres)
                                                                for snap in tqdm(snap_list))

    print('done with computation, now dumping to file')

    for i,snap in tqdm(enumerate(snap_list)):
        snap_key = 'snapshot_'+"{:03d}".format(snap)
        xy, time = out[i]
        
        for i, pt in enumerate(parttype):
            f['PartType'+str(pt)+'/xy'].create_dataset(snap_key, data=xy[i])

            f['PartType'+str(pt)+'/xy/'+snap_key].attrs['Time'] = time

    if len(snap_list) > 0:
        f.attrs['maxsnap'] = np.max(snap_list)

    f.close()

if __name__ == '__main__':
    path = '../../runs/fid-dispPoly-fg0.1/lvl5'
    name = 'fid-dispPoly-fg0.1-lvl5'

    construct_update_projection_hdf5(name, path, 2)
