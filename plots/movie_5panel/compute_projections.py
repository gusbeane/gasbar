import arepo
import numpy as np 
import h5py as h5

def make_projection_snap(path, snapnum, parttype=[0, 2, 3, 4], 
                         center=np.array([200, 200, 200]), width=30., nres=256):

    sn = arepo.Snapshot(path+'/output', snapnum, parttype=parttype, 
                        combineFiles=True, fields=['Coordinates', 'Masses'])

    range_xy = [[center[0] - width/2.0, center[0] + width/2.0], [center[1] - width/2.0, center[1] + width/2.0]]
    range_xz = [[center[0] - width/2.0, center[0] + width/2.0], [center[2] - width/2.0, center[2] + width/2.0]]
    range_yz = [[center[1] - width/2.0, center[1] + width/2.0], [center[2] - width/2.0, center[2] + width/2.0]]

    print(range_xy)

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
        heatmap_yz_out.append(heatmap_yz)
        heatmap_xz_out.append(heatmap_xz)

    return heatmap_xz, heatmap_yz, heatmap_xz

