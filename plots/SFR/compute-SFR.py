import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import astropy.units as u
import pickle

time_conv = (1 * u.kpc/(u.km/u.s)).to_value(u.Myr)

def compute_SFR(path, name, output_dir='data/'):
    # get the last snapshot
    files = glob.glob(path+'/output/snapdir_*/snapshot_*.0.hdf5')
    indices = np.array([int(re.findall('\d?\d\d\d', f)[-1]) for f in files])
    last_key = np.argmax(indices)
    file_last = files[last_key]

    last_snap = arepo.Snapshot(file_last, combineFiles=True)

    star_birthtime = last_snap.GFM_StellarFormationTime * time_conv
    star_mini = last_snap.GFM_InitialMass.as_unit(arepo.u.msol).value

    pickle.dump((star_birthtime, star_mini), open(output_dir+'SFR_'+name+'.p', 'wb'))

if __name__ == '__main__':

    basepath = '../../runs/'

    nbody = 'fid-Nbody/'
    wet = 'fid-wet/'
    fid = 'fid/'
    fid_rdisk = 'fid-disp1.0-resetDisk/'
        
    path_list = [fid+'lvl5',
                 fid+'lvl4',
                 fid+'lvl3',
                 fid_rdisk + 'lvl5']

    name_list = ['fid-lvl5',
                 'fid-lvl4',
                 'fid-lvl3',
                 'fid-disp1.0-resetDisk-lvl5']
    
    for path, name in zip(tqdm(path_list), name_list):
        out = compute_SFR(path, name)
    
