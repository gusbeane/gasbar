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
import shutil

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

def main_bar_angle(dat, Rbin = 5, firstkey = 150, nmax = 10):
    # try loading snapshot
#     dat = h5.File(fname, mode='r')
    out = {}

    keys = get_sorted_keys(dat)
    time, R, phi = get_A2_angle(dat, keys, Rbin)
    bar_angle = get_bar_angle(phi, firstkey)

    pattern_speed = np.gradient(bar_angle, time) / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    out['time'] = time
    out['firstkey'] = firstkey
    out['R'] = R
    out['phi'] = phi
    out['bar_angle'] = bar_angle
    out['pattern_speed'] = pattern_speed

    return out

@njit
def compute_apoapses(orbit):

    # first find the apoapses
    N = len(orbit)
    Rsq = np.zeros(N)
    phi = np.zeros(N)
    z = np.zeros(N)
    for i in range(N):
        Rsq[i] = orbit[i][0]**2 + orbit[i][1]**2
        phi[i] = np.arctan2(orbit[i][1], orbit[i][0])
        if phi[i] > np.pi:
            phi[i] -= 2.*np.pi

        z[i] = orbit[i][2] 

    is_apoapse_R = np.zeros(N)
    is_apoapse_phi = np.zeros(N)
    is_apoapse_z = np.zeros(N)
    
    for i in range(1, N-1):
        if Rsq[i] > Rsq[i-1] and Rsq[i] > Rsq[i+1]:
            is_apoapse_R[i] = 1
        
        if phi[i-1] < 0.0 and phi[i] > 0.0 and np.abs(phi[i])<np.pi/2.0:
            is_apoapse_phi[i] = 1
        
        if phi[i-1] > 0.0 and phi[i] < 0.0 and np.abs(phi[i])<np.pi/2.0:
            is_apoapse_phi[i] = -1
        
        if z[i] > z[i-1] and z[i] > z[i+1]:
            is_apoapse_z[i] = 1
    
    key_apo_R = np.where(is_apoapse_R==1)[0]
    key_apo_phi = np.where(np.logical_or(is_apoapse_phi==1, is_apoapse_phi==-1))[0]
    key_apo_z = np.where(is_apoapse_z==1)[0]
    
    key_apo_phi_pos = np.where(is_apoapse_phi==1)[0]
    key_apo_phi_neg = np.where(is_apoapse_phi==-1)[0]

    return key_apo_R, key_apo_phi, key_apo_z, key_apo_phi_pos, key_apo_phi_neg

@njit
def compute_freq(t, N):
    f = np.zeros(N)
    for i in range(2, N-2):
        dt = t[i+2] - t[i-2]
        f[i] = 2.*np.pi*4./dt
    
    f[0] = f[2]
    f[0] = f[1]
    f[N-1] = f[N-3]
    f[N-2] = f[N-3]

    return f

@njit
def compute_phi_freq(t, N, tneg, M):
    f = np.zeros(N)
    for i in range(2, N-2):
        dt = t[i+2] - t[i-2]
        f[i] = 2.*np.pi*4./dt

        for j in range(M):
            if t[i] == tneg[j]:
                f[i] = -f[i]
                break

    f[0] = f[2]
    f[0] = f[1]
    f[N-1] = f[N-3]
    f[N-2] = f[N-3]

    return f

def find_apoapses_compute_freq(orbit, tlist, indices):
    key_apo_R, key_apo_phi, key_apo_z, key_apo_phi_pos, key_apo_phi_neg = compute_apoapses(orbit)
    
    t_apo_R = tlist[key_apo_R]
    t_apo_phi = tlist[key_apo_phi]
    t_apo_z = tlist[key_apo_z]

    t_apo_phi_neg = tlist[key_apo_phi_neg]


    # make phi freqs negative if the apoapse crossing was negative

    # print(len(key_apo_R), len(key_apo_phi), len(key_apo_z))

    if ((len(key_apo_R) > 8) and (len(key_apo_phi) > 8) and (len(key_apo_z) > 8)):
        freq_apo_R = compute_freq(t_apo_R, len(key_apo_R))
        freq_apo_phi = compute_phi_freq(t_apo_phi, len(key_apo_phi), t_apo_phi_neg, len(t_apo_phi_neg))
        freq_apo_z = compute_freq(t_apo_z, len(key_apo_z))

        fR = np.interp(tlist, t_apo_R, freq_apo_R)
        fphi = np.interp(tlist, t_apo_phi, freq_apo_phi)
        fz = np.interp(tlist, t_apo_z, freq_apo_z)
    else:
        fR = np.full(len(tlist), np.nan)
        fphi = np.full(len(tlist), np.nan)
        fz = np.full(len(tlist), np.nan)
    
    ans = np.transpose([fR, fphi, fz])

    return ans


def loop_freq(pos, tlist, idx_list):
    N = pos.shape[1]
    ans = []
    
    for i in range(N):
        out = find_apoapses_compute_freq(pos[:,i,:], tlist, idx_list)
        ans.append(out)

    return np.array(ans)

@njit
def find_single_resonance(OmR, Omphi, Omz, Omp, lRlist, lphilist, lzlist, tol):
    best = 1E99
    best_l = (np.nan, np.nan, np.nan)
    
    for lR in lRlist:
        for lphi in lphilist:
            for lz in lzlist:
                merit = (lR * OmR + lphi * Omphi + lz*Omz) / Omp - 1.0
                if np.abs(merit) < best:
                    best = np.abs(merit)
                    best_l = (lR, lphi, lz)

    if best < tol:
        return best_l
    else:
        return (np.nan, np.nan, np.nan)

@njit
def find_resonances(Om, Omp):
    N = len(Om)
    res = np.zeros((N, 3))

    lRmin = lphimin = lzmin = -5
    lRmax = lphimax = lzmax = 5
    tol = 0.05
    
    lRlist = np.arange(lRmin, lRmax+1)
    lphilist = np.arange(lphimin, lphimax+1)
    lzlist = np.arange(lzmin, lzmax+1)
    
    for i in range(N):
        OmR = Om[i][0]
        Omphi = Om[i][1]
        Omz = Om[i][2]
        
        this_res = find_single_resonance(OmR, Omphi, Omz, Omp, lRlist, lphilist, lzlist, tol)
        res[i][0] = this_res[0]
        res[i][1] = this_res[1]
        res[i][2] = this_res[2]
    
    return res

def loop_res(freq, pspeed):
    Nsnap = freq.shape[1]
    res = []

    for i in range(Nsnap):
        if pspeed[i] > 0.0:
            res.append(find_resonances(freq[:,i,:], pspeed[i]))
        else:
            res.append(np.full(np.shape(freq[:,i,:]), np.nan))
    
    return np.array(res)

def _run_chunk(name, chunk_idx, prefix, phase_space_path, center, indices, pattern_speed):
    fin = phase_space_path + name + '/phase_space_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5in = h5.File(fin, mode='r')
    
    indices = np.array(h5in['Time'])
    indices = np.arange(len(indices))

    # to be copied to later
    fout = prefix + 'freq_' + name + '.' + str(chunk_idx) + '.hdf5'

    tmpout = '/scratch/' + 'freq_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5out = h5.File(tmpout, mode='w')

    tlist = np.array(h5in['Time'])

    # halo particles
    pos = np.array(h5in['PartType1/Coordinates'])
    ids = np.array(h5in['PartType1/ParticleIDs'])
    pos -= center
        
    ans = loop_freq(pos, tlist, indices)
    res = loop_res(ans, pattern_speed)
    h5out.create_dataset('PartType1/Frequencies', data=ans)
    h5out.create_dataset('PartType1/Resonances', data=res)
    h5out.create_dataset('PartType1/ParticleIDs', data=ids)

    # load disk particles
    pos = np.array(h5in['PartType2/Coordinates'])
    ids = np.array(h5in['PartType2/ParticleIDs'])
    pos -= center
        
    ans = loop_freq(pos, tlist, indices)
    res = loop_res(ans, pattern_speed)
    h5out.create_dataset('PartType2/Frequencies', data=ans)
    h5out.create_dataset('PartType2/Resonances', data=res)
    h5out.create_dataset('PartType2/ParticleIDs', data=ids)

    # load bulge particles
    pos = np.array(h5in['PartType3/Coordinates'])
    ids = np.array(h5in['PartType3/ParticleIDs'])
    pos -= center
        
    ans = loop_freq(pos, tlist, indices)
    res = loop_res(ans, pattern_speed)
    h5out.create_dataset('PartType3/Frequencies', data=ans)
    h5out.create_dataset('PartType3/Resonances', data=res)
    h5out.create_dataset('PartType3/ParticleIDs', data=ids)

    # load star particles (if they exist)
    if 'PartType4' in h5in.keys():
        pos = np.array(h5in['PartType4/Coordinates'])
        ids = np.array(h5in['PartType4/ParticleIDs'])

        pos -= center
        
        ans = loop_freq(pos, tlist, indices)
        res = loop_res(ans, pattern_speed)
        h5out.create_dataset('PartType4/Frequencies', data=ans)
        h5out.create_dataset('PartType4/Resonances', data=res)
        h5out.create_dataset('PartType4/ParticleIDs', data=ids)

    # for j in range(len(ans)):
    #     h5out.create_dataset('bar_metrics/'+str(ids[j]), data=ans[j])
    # h5out.create_dataset('freqs', data=ans)
    
    h5out.create_dataset('tlist', data=tlist)
    # h5out.create_dataset('id_list', data=ids)
    h5out.create_dataset('idx_list', data=indices)

    # bar_angle = np.mod(bar_angle_out['bar_angle'][indices], 2.*np.pi)
    # h5out.create_dataset('bar_angle', data=bar_angle)

    h5out.close()
    h5in.close()

    # copy from local storage to actual output directory
    shutil.copy(tmpout, fout)
    os.remove(tmpout)

    return 0

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

def run(path, name, nsnap, nproc, phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data/'):
    prefix = 'data/freq_' + name +'/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # get some preliminary variables
    center, firstkey, indices = preprocess_center(name)
    fourier = read_fourier(name)
    fourier_out = main_bar_angle(fourier, firstkey=firstkey)
    pspeed = fourier_out['pattern_speed']
    
    # do standard fourier and bar angle stuff

    nchunk = len(glob.glob(phase_space_path+name+'/phase_space_'+name+'.*.hdf5'))
    print(nchunk)
    # tot_ids = []
    _ = Parallel(n_jobs=nproc) (delayed(_run_chunk)(name, i, prefix, phase_space_path, center, indices, pspeed) for i in tqdm(range(nchunk)))
        
    # for i in tqdm(range(nchunk)):
        # print(i)
        # _run_chunk(name, i, prefix, phase_space_path, center, indices)

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../runs/'

    Nbody = 'Nbody'
    phgvS2Rc35 = 'phantom-vacuum-Sg20-Rc3.5'

    pair_list = [(Nbody, 'lvl4'), (Nbody, 'lvl3'),
                 (phgvS2Rc35, 'lvl4'), (phgvS2Rc35, 'lvl3'),
                 (phgvS2Rc35, 'lvl3-rstHalo')]

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
