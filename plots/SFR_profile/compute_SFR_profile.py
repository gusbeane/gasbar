import numpy as np
import arepo
import sys
from tqdm import tqdm
import pickle

def my_bin(X, Xbins, Y, func=np.mean):
    Xmin = Xbins[0]
    Xmax = Xbins[-1]
    assert Xmin==np.min(Xbins), "First entry of Xbins is not the min value"
    assert Xmax==np.max(Xbins), "Last entry of Xbins is not the max value"

    keys = np.logical_and(X > Xmin, X < Xmax)

    digit = np.digitize(X[keys], Xbins)
    X_avg = [X[keys][digit == i].mean() for i in range(1, len(Xbins))]
    Y_func = [func(Y[keys][digit == i]) for i in range(1, len(Xbins))]

    return np.array(X_avg), np.array(Y_func)

def compute_SFR_profile(path, snapnum, name, output_dir='data/', center=[200, 200, 200],
                        Rmin=0.1, Rmax=20, nbins=20):
    # try loading snapshot
    sn = arepo.Snapshot(path+'/output/', snapnum, combineFiles=True)

    #get pos of gas
    pos = np.subtract(sn.part0.pos, center)
    R = np.linalg.norm(pos[:,:2], axis=1)
    sfr = np.copy(sn.part0.sfr.value)

    #do binning
    # bins = np.logspace(np.log10(Rmin), np.log10(Rmax), nbins)
    bins = np.linspace(Rmin, Rmax, nbins)
    R_binned, tot_sfr_binned = my_bin(R, bins, sfr, np.sum)

    # now go through and divide by area of bin
    for i in range(len(R_binned)):
        surf_area = np.pi * (bins[i+1]**2 - bins[i]**2)
        tot_sfr_binned /= surf_area

    pickle.dump((R_binned, tot_sfr_binned), open(output_dir+'SFR_profile_'+name+'_snap'+str(snapnum)+'.p', 'wb'))


if __name__ == '__main__':

    basepath = '../../runs/'

    fid_g1 = 'fid-disp1.0-fg0.1'

    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        snapnum_list = [10, 50, 100]#, 100, 150, 200, 300, 400, 500, 600]
        pair_list = [(fid_g1, 'lvl5')]
    else:
        snapnum_list = [10, 50, 100, 100, 150, 200, 300, 400, 500, 600]
        pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4')]#, (fid_g1, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                            
    # nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    # if len(sys.argv) == 3:
    #     i = int(sys.argv[2])
    #     path = path_list[i]
    #     name = name_list[i]
    #     nsnap = nsnap_list[i]

    #     run(path, name, nsnap)
    # else:
    #     for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
    #         run(path, name, nsnap)


    
    # snapnum_list = [10, 50, 100, 150, 200, 300, 400, 500, 600]

    for path, name in zip(tqdm(path_list), name_list):
        for snapnum in snapnum_list:
            out = compute_SFR_profile(path, snapnum, name)
    
