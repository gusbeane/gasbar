import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import pickle

def compute_L4L5(path, nsnap, name, Rguess=3, search_radius=0.7, nmax=100, npoly=5,
                 bar_angle_data='../bar_angle/data/', output_dir='data/'):
    # setup output lists
    snap_index_list = []
    time_list = []
    L4_list = []
    L5_list = []

    fname = bar_angle_data + 'bar_angle_' + name + '.p'
    bar_data = pickle.load(open(fname, 'rb'))

    ba, ps = bar_data['poly_eval'][npoly]
    first_ba = ba[0]

    for this_idx, this_ba in zip(tqdm(range(nsnap)), ba):

        sn = arepo.Snapshot(path+'output/', this_idx, combineFiles=True)
        time = sn.header.time.as_unit(3.1536E13 * arepo.u.second).value # in Myr

        snap_index_list.append(this_idx)
        time_list.append(time)

        # if we're not past the first bar angle
        if np.abs(this_ba - first_ba) < 0.01:
            L4_list.append([0, 0, 0])
            L5_list.append([0, 0, 0])
            continue

        first = True
        for i,npart in enumerate(sn.NumPart_Total):
            if i==0 or npart==0:
                continue
            part = getattr(sn, 'part'+str(i))

            if first:
                pos = part.pos
                pot = part.pot
                ids = part.id
            else:
                pos = np.concatenate((pos, part.pos))
                pot = np.concatenate((pot, part.pot))
                ids = np.concatenate((ids, part.id))


        L4, L5 = _get_L4L5_(pos, pot, ids, nmax, this_ba, search_radius, Rguess)

        L4_list.append(L4)
        L5_list.append(L5)

    out = {}
    out['snap_index'] = snap_index_list
    out['time'] = np.array(time_list)
    out['L4'] = np.array(L4_list)
    out['L5'] = np.array(L5_list)

    pickle.dump(out, open(output_dir + 'L4L5_'+name+'.p', 'wb'))

    return out

def _get_L4L5_(pos, pot, ids, nmax, ba, search_radius, Rguess):
    # spin the Galaxy like a record player
    R = np.sqrt(np.add(np.square(pos[:,0]), np.square(pos[:,1])))
    phi = np.arctan2(pos[:,1], pos[:,0])

    phinew = np.subtract(phi, ba)
    Xnew = np.multiply(R, np.cos(phinew))
    Ynew = np.multiply(R, np.sin(phinew))

    pos[:,0] = Xnew
    pos[:,1] = Ynew

    guess_L4 = np.array([0, Rguess, 0])
    guess_key = None    
    # now begin the search
    L4 = np.array([0, 0, 0])
    for _ in range(nmax):
        old_guess_key = np.copy(guess_key)
        dist = np.linalg.norm(np.subtract(pos, guess_L4), axis=1)
        keys = dist < search_radius

        k = np.argmin(dist)

        guess_key = np.argmin(pot[keys])
        guess_L4 = pos[guess_key]
        if guess_key == old_guess_key:
            L4 = np.copy(guess_L4)
            break

    guess_L5 = np.array([0, -Rguess, 0])
    L5 = np.array([0, 0, 0])
    for _ in range(nmax):
        old_guess_key = np.copy(guess_key)
        dist = np.linalg.norm(np.subtract(pos, guess_L5), axis=1)
        keys = dist < search_radius
        
        guess_key = np.argmin(pot[keys])
        guess_L5 = pos[guess_key]
        if guess_key == old_guess_key:
            L5 = np.copy(guess_L5)
            break

    return L4, L5


if __name__ == '__main__':

    basepath = '../../runs/'

    nbody = 'fid-Nbody/'
    wet = 'fid-wet/'
    fid = 'fid/'
    
    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        path_list = [basepath + nbody + 'lvl5/']
        name_list = ['nbody-lvl5']
    else:
        lvl_list = [5, 4, 3]
        path_list = [basepath + nbody + 'lvl' + str(i) + '/' for i in lvl_list]
        name_list = ['nbody-lvl' + str(i) for i in lvl_list]

    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        nsnap = len(glob.glob(path_list[i]+'/output/snapdir*/*.0.hdf5'))
        out = compute_L4L5(path_list[i], nsnap, name_list[i])
    else:
        for path, name in zip(tqdm(path_list), name_list):
            nsnap = len(glob.glob(path+'/output/snapdir*/*.0.hdf5'))
            out = compute_L4L5(path, nsnap, name)
    
