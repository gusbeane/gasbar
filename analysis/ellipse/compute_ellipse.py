import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import os
import pickle
import h5py as h5
from numba import njit

from photutils.isophote import EllipseGeometry, Ellipse

from joblib import Parallel, delayed

nres = 512
rng = [[-10, 10], [-10, 10]]

def get_surfdens_map(sn, center):
    pos = sn.part2.pos.value - center
    if sn.NumPart_Total[4] > 0:
        pos = np.concatenate((pos, sn.part4.pos.value - center))

    x = pos[:,0]
    y = pos[:,1]
    heatmap_xy, _, _ = np.histogram2d(x, y, bins=(nres, nres), range=rng)
    
    return heatmap_xy.T


def get_center(name, sn):
    
    pot_min_1 = np.min(sn.part1.pot.value)
    pot_min_2 = np.min(sn.part2.pot.value)
    pot_min_3 = np.min(sn.part3.pot.value)
    
    center_1 = sn.part1.pos.value[np.argmin(sn.part1.pot.value)]
    center_2 = sn.part2.pos.value[np.argmin(sn.part2.pot.value)]
    center_3 = sn.part3.pos.value[np.argmin(sn.part3.pot.value)]
    
    center = center_1
    pot_min = pot_min_1
    
    if pot_min_2 < pot_min:
        center = center_2
        pot_min = pot_min_2
        
    if pot_min_3 < pot_min:
        center = center_3
        pot_min = pot_min_3
    
    return center

def get_bar_length(isolist):
    k = np.argmax(isolist.eps)
    ellip = np.max(isolist.eps)
    Rbar_e = isolist.sma[k] * 20/nres
    
    kpa = k
    for i in range(len(isolist)-k):
        kpa += 1
        delta_PA = np.abs(isolist[kpa].pa - isolist[k].pa) * 180/np.pi
        print(delta_PA)
        if delta_PA > 5:
            kpa -= 1
            break
    
    Rbar_PA = isolist.sma[kpa] * 20/nres
    
    return Rbar_e, Rbar_PA, ellip

def do_ellipse_fit(data):
    geometry = EllipseGeometry(x0=nres/2, y0=nres/2, sma=80, eps=0.5, 
                               pa=20.0 * np.pi / 180.0)
    ellipse = Ellipse(data)
    
    try:
        isolist = ellipse.fit_image(step=1.0, minsma=10., maxsma=256, linear=True)
    
        Rbar_e, Rbar_PA, ellip = get_bar_length(isolist)
    
    except:
        Rbar_e = np.nan
        Rbar_PA = np.nan
        ellip = np.nan
    
    return Rbar_e, Rbar_PA, ellip

def _runner(path, name, snap, ptypes=[1, 2, 3, 4]):
    sn = arepo.Snapshot(path + '/output/', snap, 
                        parttype=ptypes, 
                        fields=['Coordinates', 'Velocities', 'Potential'],
                        combineFiles=True)
    
    center = get_center(name, sn)
    
    data = get_surfdens_map(sn, center)
    
    Rbar_e, Rbar_PA, ellip = do_ellipse_fit(data)
    
    Time = sn.Time.value
    
    
    # Package it all together
    output = (Rbar_e, Rbar_PA, ellip, Time)
    
    return output

def run(path, name, nproc):
    
    nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
    
    out = Parallel(n_jobs=nproc) (delayed(_runner)(path, name, i) for i in tqdm(range(nsnap)))

    Rbar_e = np.array([out[i][0] for i in range(len(out))])
    Rbar_PA = np.array([out[i][1] for i in range(len(out))])
    Ellipticity         = np.array([out[i][2] for i in range(len(out))])
    Time         = np.array([out[i][3] for i in range(len(out))])

    out = {'Rbar_e' : Rbar_e,
           'Rbar_PA': Rbar_PA,
           'Ellipticity': Ellipticity,
           'Time'          : Time}
    
    np.save('ellipse_'+name+'.npy', out)

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../'

    Nbody = 'Nbody'
    SMUGGLE = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl3', 0), # 0
                 (SMUGGLE, 'lvl3', 0), # 8
                ]


    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + 'runs/' + p[0] + '/' + p[1] for p in pair_list]
    Nang_list  = [p[2] for p in pair_list]
    # ic_list   = [basepath + 'ics/' + p[0] + '/' + p[1] for p in pair_list]
    
  
    i = int(sys.argv[2])
    path = path_list[i]
    name = name_list[i]
    Nang = Nang_list[i]

    if Nang > 0:
        out = run_ang(path, name, nproc, Nang)
    else:
        out = run(path, name, nproc)
