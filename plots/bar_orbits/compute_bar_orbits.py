import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import h5py as h5
import glob
import os
from numba import njit
import re
from sklearn.cluster import KMeans

from joblib import Parallel, delayed

def read_fourier(name, basepath='/n/home01/abeane/starbar/plots/'):
    f = h5.File(basepath+'/fourier_component/data/fourier_' + name + '.hdf5', mode='r')
    return f

def read_snap(path, idx, parttype=[0], fields=['Coordinates', 'Masses', 'Velocities', 'ParticleIDs']):
    
    fname = path + '/output'
    
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

def get_bar_angle(phi, firstkey):
    out = np.zeros(len(phi))

    # set the first bar angle
    first_bar_angle = phi[firstkey]/2.0
    out[firstkey] = first_bar_angle
    
    # set all subsequent angles
    for i in np.arange(firstkey+1, len(out)):
        dphi = phi[i] - phi[i-1]
        if dphi < -np.pi:
            dphi += 2.*np.pi
        out[i] = out[i-1] + dphi/2.0

    # set all previous angles to be the bar angle
    for i in np.arange(0, firstkey):
        out[i] = first_bar_angle

    return out

def get_sorted_keys(dat):
    keys = list(dat.keys())
    # only keep keys that are snapshot keys
    keys = [k for k in keys if 'snapshot' in k]

    # extract and sort indices
    indices = [int(re.findall(r'\d?\d?\d\d\d', k)[0]) for k in keys]
    sorted_arg = np.argsort(indices)
    keys_sorted = [keys[i] for i in sorted_arg]

    return keys_sorted

def get_A2_angle(dat, keys, Rbin):
    Rlist = np.array([np.array(dat[k]['Rlist']) for k in keys])
    A2r = np.array([np.array(dat[k]['A2r']) for k in keys])
    A2i = np.array([np.array(dat[k]['A2i']) for k in keys])
    phi = np.arctan2(A2i, A2r)
    phi = phi[:,Rbin]
    R_at_Rbin = Rlist[:,Rbin]
    
    time = np.array(dat['time'])

    return time, R_at_Rbin, phi

def get_bar_length(dat, keys, Rmin=2, Rmax=10, ratio_cut = 2):
    Rlist = np.array([np.array(dat[k]['Rlist']) for k in keys])

    A0 = np.array([np.array(dat[k]['A0']) for k in keys])
    A1r = np.array([np.array(dat[k]['A1r']) for k in keys])
    A1i = np.array([np.array(dat[k]['A1i']) for k in keys])
    A2r = np.array([np.array(dat[k]['A2r']) for k in keys])
    A2i = np.array([np.array(dat[k]['A2i']) for k in keys])
    A3r = np.array([np.array(dat[k]['A3r']) for k in keys])
    A3i = np.array([np.array(dat[k]['A3i']) for k in keys])
    A4r = np.array([np.array(dat[k]['A4r']) for k in keys])
    A4i = np.array([np.array(dat[k]['A4i']) for k in keys])
    A5r = np.array([np.array(dat[k]['A5r']) for k in keys])
    A5i = np.array([np.array(dat[k]['A5i']) for k in keys])
    A6r = np.array([np.array(dat[k]['A6r']) for k in keys])
    A6i = np.array([np.array(dat[k]['A6i']) for k in keys])
    
    I0 = A0/2.
    I1 = np.sqrt(A1r*A1r + A1i*A1i)
    I2 = np.sqrt(A2r*A2r + A2i*A2i)
    I3 = np.sqrt(A3r*A3r + A3i*A3i)
    I4 = np.sqrt(A4r*A4r + A4i*A4i)
    I5 = np.sqrt(A5r*A5r + A5i*A5i)
    I6 = np.sqrt(A6r*A6r + A6i*A6i)
    
    Ib = I0 + I2 + I4 + I6
    Iib = I0 - I2 + I4 - I6
    
    IbIib = Ib/Iib
    
    Rbar_list = []
    for i,k in enumerate(keys):
        R = Rlist[i,:]
        ratio = IbIib[i,:]
        
        Rkey = np.logical_and(R > Rmin, R< Rmax)
        ratio = ratio[Rkey]
        R = R[Rkey]
        j = 0
        try:
            while ratio[j] > ratio_cut:
                j += 1
            Rbar = R[j-1] + (ratio_cut - ratio[j-1]) * (R[j]-R[j-1])/(ratio[j]-ratio[j-1])
        except:
            Rbar = np.nan
        Rbar_list.append(Rbar)

    time = np.array(dat['time'])    
    
    return time, np.array(Rbar_list)

def evaluate_polynomial(pfit, n, time, bar_angle_firstkey, firstkey):
    pfit_n = pfit[n]
    poly_bar_angle = np.zeros(len(time))
    poly_pattern_speed = np.zeros(len(time))

    for i in range(n+1):
        ba = pfit_n[i] * time ** (n-i)
        poly_bar_angle[firstkey:] += ba[firstkey:]
        ps = (n-i) * pfit_n[i] * time**(n-1-i)
        poly_pattern_speed[firstkey:] += ps[firstkey:]

    poly_bar_angle[:firstkey] += bar_angle_firstkey

    poly_pattern_speed = poly_pattern_speed / u.Myr
    poly_pattern_speed = poly_pattern_speed.to_value(u.km/u.s/u.kpc)

    return poly_bar_angle, poly_pattern_speed

def main_bar_angle(dat, Rbin = 5, firstkey = 150, nmax = 10):
    out = {}

    keys = get_sorted_keys(dat)
    time, R, phi = get_A2_angle(dat, keys, Rbin)
    time, Rbar = get_bar_length(dat, keys)
    #     Rlist, Iibar = get_bar_length(dat, keys)
    bar_angle = get_bar_angle(phi, firstkey)

    pattern_speed = np.gradient(bar_angle, time) / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    pfit = [np.polyfit(time[firstkey:], bar_angle[firstkey:], i) for i in range(nmax)]
    
    out['time'] = time
    out['firstkey'] = firstkey
    out['R'] = R
    out['Rbar'] = Rbar
    out['phi'] = phi
    out['bar_angle'] = bar_angle
    out['pattern_speed'] = pattern_speed
    out['pfit'] = pfit

    # now evaluate the polynomial for each fit and save the result
    out['poly_eval'] = {}
    for n in range(nmax):
        poly_bar_angle, poly_pattern_speed = evaluate_polynomial(pfit, n, time, bar_angle[firstkey], firstkey)

        out['poly_eval'][n] = (poly_bar_angle, poly_pattern_speed)

    return out

# rotate wlist
def rotate_w(w, ang):

    Rmat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang), np.cos(ang),  0.0],
                     [0.0,         0.0,          1.0]])
    w = np.swapaxes(w, 0, 1)

    w[:3,:] = np.matmul(Rmat, w[:3,:])
    w[3:,:] = np.matmul(Rmat, w[3:,:])

    w = np.swapaxes(w, 0, 1)
    return w

# rotate by bar angle at each time step

def rotate_wlist(wlist, bar_angle_out, idx_list):
    bar_angle = np.mod(bar_angle_out['bar_angle'][idx_list], 2.*np.pi)

    Rwlist = np.zeros(np.shape(wlist))

    for i,idx in enumerate(idx_list):
        Rwlist[i] = rotate_w(wlist[i], -bar_angle[i])
    
    return Rwlist

# custom kmeans
@njit
def k_means_2(pts):
    # first choose the center
    N = len(pts)
    cen0 = pts[np.random.choice(N, 1)][0]
    
    # now compute the distance squared from the pts to the center
    dist = np.zeros(N)
    for i in range(N):
        dist[i] = (cen0[0]-pts[i][0])**2 + (cen0[1]-pts[i][1])**2

    # choose second center with probability dist^2
    p = dist / np.sum(dist)
    p_ = np.random.rand()
    for i in range(N):
        p_ -= p[i]
        if p_ < 0.0:
            break
    cen1 = pts[i]
    
    # now proceed with kmeans
    keep_going = True
    
    group_old = np.zeros(N)
    group_new = np.zeros(N)
    while keep_going:
        # first compute the new centers based on the assignments
        cen0_new = np.zeros(2)
        cen1_new = np.zeros(2)
        N0 = 0
        N1 = 0
        for i in range(N):
            d0 = (cen0[0]-pts[i][0])**2 + (cen0[1]-pts[i][1])**2
            d1 = (cen1[0]-pts[i][0])**2 + (cen1[1]-pts[i][1])**2
            if d0 < d1:
                group_new[i] = 0
                cen0_new += pts[i]
                N0 += 1
            else:
                group_new[i] = 1
                cen1_new += pts[i]
                N1 += 1
    
        cen0_new /= N0
        cen1_new /= N1
    
        # check to see if there were any reassignments
        no_reassign = True
        for i in range(N):
            if group_new[i] != group_old[i]:
                no_reassign=False
                break
    
        # if there were no reassignments, then we end. otherwise iterate again
        if no_reassign:
            keep_going = False
        
        group_old = group_new
        cen0 = cen0_new
        cen1 = cen1_new
    
    key0 = group_new==0
    key1 = group_new==1
    
    return pts[key0], pts[key1]

@njit
def compute_angle_from_xaxis(pos):

    phi = np.arctan2(pos[:,1], pos[:,0])

    # assume pos x
    key = phi > np.pi
    phi_p = np.copy(phi)
    phi_p[key] = phi[key] - 2.*np.pi

    phi_n = np.copy(phi) + np.pi
    key = phi_n > np.pi
    phi_n[key] = phi_n[key] - 2.*np.pi

    phi_ = np.minimum(np.abs(phi_p), np.abs(phi_n))

    return phi_

@njit
def compute_apoapses(orbit):

    # first find the apoapses
    N = len(orbit)
    rsq = np.zeros(N)
    for i in range(N):
        rsq[i] = orbit[i][0]**2 + orbit[i][1]**2 + orbit[i][2]**2

    is_apoapse = np.zeros(N)
    
    for i in range(1, N-1):
        if rsq[i] > rsq[i-1] and rsq[i] > rsq[i+1]:
            is_apoapse[i] = 1
    
    key_apo = np.where(is_apoapse==1)[0]
    apo = orbit[key_apo]
    
    return key_apo, apo

@njit
def compute_trapping_metrics(apo0, apo1, dt):
    # now compute the four metrics from PWK19a
    phi_0 = compute_angle_from_xaxis(apo0)
    phi_1 = compute_angle_from_xaxis(apo1)
    
    # get max of delta phibar
    ave_delta_phibar = np.maximum(np.mean(phi_0), np.mean(phi_1))
    
    # average of std in R
    R_0 = np.zeros(len(apo0))
    R_1 = np.zeros(len(apo1))
    for i in range(len(apo0)):
        R_0[i] = np.sqrt(apo0[i][0]**2 + apo0[i][1]**2)
    for i in range(len(apo1)):
        R_1[i] = np.sqrt(apo1[i][0]**2 + apo1[i][1]**2)
    
    std_R0 = np.std(R_0)
    std_R1 = np.std(R_1)
    ave_stdR = (std_R0+std_R1)/2.0
    
    mean_R0 = np.mean(R_0)
    mean_R1 = np.mean(R_1)
    mean_R = (mean_R0 + mean_R1)/2.0
    
    std_phi0 = np.std(phi_0)
    std_phi1 = np.std(phi_1)
    ave_stdphi = (std_phi0+std_phi1)/2.0
    
    omega_r = np.abs(dt)
    omega_r = 1./omega_r
    
    return ave_delta_phibar, ave_stdR, mean_R, ave_stdphi, omega_r

@njit
def find_apoapses_do_kmeans(orbit, tlist, indices):
    key_apo, apo = compute_apoapses(orbit)
    
    t_apo = np.zeros(len(key_apo))
    idx_apo = np.zeros(len(key_apo))
    for i in range(len(key_apo)):
        t_apo[i] = tlist[key_apo[i]]
        idx_apo[i] = indices[key_apo[i]]
    
    trap_met_list = np.zeros((len(key_apo), 6))
    
    for i,t in enumerate(t_apo):
        trap_met_list[i][0] = idx_apo[i]
        
        key_sort = np.argsort(np.abs(t_apo - t))
        if len(key_sort) < 20:
            continue
        key_closest = key_sort[0]
        key_second_closest = key_sort[1]
        dt = t_apo[key_closest] - t_apo[key_second_closest]

        key_20_closest = key_sort[:20]    
        apo_20_closest = apo[key_20_closest]
        
        apo0, apo1 = k_means_2(apo_20_closest)
        
        trap_met = compute_trapping_metrics(apo0, apo1, dt)
        
        for j in range(1, 6):
            trap_met_list[i][j] = trap_met[j-1]
    
    return trap_met_list

def loop_trapping_metrics(Rwlist, tlist, idx_list):
    N = Rwlist.shape[1]
    ans = []
    
    for i in range(N):
        out = find_apoapses_do_kmeans(Rwlist[:,i,:3], tlist, idx_list)
        ans.append(out)

    return ans

def preprocess_center(name):
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
        firstkey=150
        indices = np.arange(nsnap)
        # indices_analyze = np.arange(500, 1100, 20)
    else:
        center = np.array([200, 200, 200])
        firstkey=0
        indices = np.arange(nsnap)
    
    return center, firstkey, indices

def _run_chunk(name, chunk_idx, prefix, phase_space_path, center, bar_angle_out, indices):
    fin = phase_space_path + name + '/phase_space_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5in = h5.File(fin, mode='r')
        
    fout = prefix + 'bar_orbit_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    tlist = np.array(h5in['Time'])
    pos = np.array(h5in['Coordinates']) - center
    vel = np.array(h5in['Velocities'])
        
    w = np.concatenate((pos, vel), axis=-1)
    w = np.swapaxes(w, 0, 1)

    Rwlist = rotate_wlist(w, bar_angle_out, indices)
    ans = loop_trapping_metrics(Rwlist, tlist, indices)

    ids = np.array(h5in['ParticleIDs'])
    # tot_ids = np.concatenate((tot_ids, ids))

    for j in range(len(ans)):
        h5out.create_dataset('bar_metrics/'+str(ids[j]), data=ans[j])
    
    h5out.create_dataset('tlist', data=tlist)
    h5out.create_dataset('id_list', data=ids)
    h5out.create_dataset('idx_list', data=indices)

    bar_angle = np.mod(bar_angle_out['bar_angle'][indices], 2.*np.pi)
    h5out.create_dataset('bar_angle', data=bar_angle)
    return None

def run(path, name, nsnap, nproc, phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data_tmp/'):
    prefix = 'data/bar_orbit_' + name +'/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # get some preliminary variables
    center, firstkey, indices = preprocess_center(name)
    
    # do standard fourier and bar angle stuff
    fourier = read_fourier(name)
    bar_angle_out = main_bar_angle(fourier, firstkey=firstkey)

    nchunk = len(glob.glob(phase_space_path+name+'/phase_space_'+name+'.*.hdf5'))
    # tot_ids = []
    _ = Parallel(n_jobs=nproc) (delayed(_run_chunk)(name, i, prefix, phase_space_path, center, bar_angle_out, indices) for i in tqdm(range(nchunk)))
        

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl4'), (Nbody, 'lvl3'),
                 (phgvS2Rc35, 'lvl4'), (phgvS2Rc35, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]

    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]

    if len(sys.argv) == 3:
        i = int(sys.argv[2])
        path = path_list[i]
        name = name_list[i]
        nsnap = nsnap_list[i]

        out = run(path, name, nsnap, nproc)
    else:
        for path, name, nsnap in zip(tqdm(path_list), name_list, nsnap_list):
            out = run(path, name, nsnap, nproc)
