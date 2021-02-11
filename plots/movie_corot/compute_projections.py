import os
import glob

import numpy as np 
import h5py as h5
from joblib import Parallel, delayed

import arepo
from tqdm import tqdm
from bar_angle import master_bar_angle

def make_projection_snap(path, snapnum, parttype=[0, 2, 3, 4], 
                         center=np.array([200, 200, 200]), width=30., nres=256,
                         corot=False, bar_angle=None, fourier_n=5):

    sn = arepo.Snapshot(path+'/output', snapnum, parttype=parttype, 
                        combineFiles=True, fields=['Coordinates', 'Masses'])

    time = sn.Time

    ba, _ = bar_angle['poly_eval'][fourier_n]
    ba = ba[snapnum]

    range_xy = [[center[0] - width/2.0, center[0] + width/2.0], [center[1] - width/2.0, center[1] + width/2.0]]
    range_xz = [[center[0] - width/2.0, center[0] + width/2.0], [center[2] - width/2.0, center[2] + width/2.0]]
    range_yz = [[center[1] - width/2.0, center[1] + width/2.0], [center[2] - width/2.0, center[2] + width/2.0]]

    surf = (width/nres)**(2.0)

    heatmap_xy_out = []
    heatmap_xz_out = []
    heatmap_yz_out = []
    for pt in parttype:
        if sn.NumPart_Total[pt] == 0:
            heatmap_xy_out.append(np.zeros((nres, nres)))
            heatmap_xz_out.append(np.zeros((nres, nres)))
            heatmap_yz_out.append(np.zeros((nres, nres)))
            continue

        part = getattr(sn, 'part'+str(pt))

        x = part.pos[:,0]
        y = part.pos[:,1]
        z = part.pos[:,2]

        if corot:
            phi = np.arctan(y, x)
            R = np.sqrt(x**2 + y**2)
            phi -= ba
            x = R * np.cos(phi)
            y = R * np.sin(phi)

        xbool = np.logical_and(x > center[0] - width/2.0, x < center[0] + width/2.0)
        ybool = np.logical_and(y > center[1] - width/2.0, y < center[1] + width/2.0)
        zbool = np.logical_and(z > center[2] - width/2.0, z < center[2] + width/2.0)

        keys = np.logical_and(np.logical_and(xbool, ybool), zbool)

        if sn.MassTable[pt] > 0:
            weights = None
            postfac = sn.MassTable[pt] / surf
        else:
            weights = part.Masses[keys] / surf
            postfac = 1.0

        heatmap_xy, _, _ = np.histogram2d(x[keys], y[keys], bins=(nres, nres), range=range_xy, weights=weights)
        heatmap_xz, _, _ = np.histogram2d(x[keys], z[keys], bins=(nres, nres), range=range_xz, weights=weights)
        heatmap_yz, _, _ = np.histogram2d(y[keys], z[keys], bins=(nres, nres), range=range_yz, weights=weights)

        heatmap_xy *= postfac
        heatmap_xz *= postfac
        heatmap_yz *= postfac

        heatmap_xy_out.append(heatmap_xy)
        heatmap_xz_out.append(heatmap_xz)
        heatmap_yz_out.append(heatmap_yz)

    return heatmap_xy_out, heatmap_xz_out, heatmap_xy_out, time

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

def construct_update_projection_hdf5(name, path, nproc=1, parttype=[0, 2, 3, 4], center=np.array([200., 200., 200.]),
                                     width=30., nres=256, output_dir='data/', corot=False, fourier_path=None, fourier_n=5,
                                     firstkey=150):

    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    assert nsnap > 0,"No output files detected for name="+name

    if corot:
        assert fourier_path is not None,"Must specify fourier_path for corot"

    fname = name
    if corot:
        fname = fname + '_corot'
    fname = fname + '_w' + "{:.01f}".format(width) + '_n' + str(nres) + '.hdf5' 

    if not do_i_continue(output_dir + '/' + fname, nsnap, parttype):
        return
    
    f = h5.File(output_dir + '/' + fname, mode='a')
    if corot:
        fourier = h5.File(fourier_path+'/fourier_'+name+'.hdf5')
        bar_angle = master_bar_angle(fourier, firstkey=firstkey)


    if 'width' not in f.attrs.keys():
        f.attrs['width'] = width
    if 'nres' not in f.attrs.keys():
        f.attrs['nres'] = nres

    for pt in parttype:
        if 'PartType' + str(pt) not in f.keys():
            f.create_group('PartType' + str(pt)+'/xy')
            f.create_group('PartType' + str(pt)+'/xz')
            f.create_group('PartType' + str(pt)+'/yz')

    

    snap_list = []
    for snap in range(nsnap):
        snap_key = 'snapshot_'+"{:03d}".format(snap)
        if snap_key not in f['PartType'+str(parttype[0])+'/xy']:
            snap_list.append(snap)

    out = Parallel(n_jobs=nproc) (delayed(make_projection_snap)(path, snap, parttype=parttype,
                                                                center=center, width=width, nres=nres,
                                                                corot=corot, bar_angle=bar_angle, 
                                                                fourier_n=fourier_n)
                                                                for snap in tqdm(snap_list))

    print('done with computation, now dumping to file')

    for i,snap in tqdm(enumerate(snap_list)):
        snap_key = 'snapshot_'+"{:03d}".format(snap)
        xy, xz, yz, time = out[i]
        
        for i, pt in enumerate(parttype):
            f['PartType'+str(pt)+'/xy'].create_dataset(snap_key, data=xy[i])
            f['PartType'+str(pt)+'/xz'].create_dataset(snap_key, data=xz[i])
            f['PartType'+str(pt)+'/yz'].create_dataset(snap_key, data=yz[i])

            f['PartType'+str(pt)+'/xy/'+snap_key].attrs['Time'] = time
            f['PartType'+str(pt)+'/xz/'+snap_key].attrs['Time'] = time
            f['PartType'+str(pt)+'/yz/'+snap_key].attrs['Time'] = time

    if len(snap_list) > 0:
        f.attrs['maxsnap'] = np.max(snap_list)

    f.close()

if __name__ == '__main__':
    path = '../../runs/fid-dispPoly-fg0.1/lvl5'
    name = 'fid-dispPoly-fg0.1-lvl5'

    construct_update_projection_hdf5(name, path, 2)
