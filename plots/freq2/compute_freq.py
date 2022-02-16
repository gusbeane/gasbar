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

def find_apoapses_compute_freq(orbit, tlist, indices, idx_analyze_list, width):
    R = np.linalg.norm(orbit[:,:2], axis=1)
    phi = np.arctan2(orbit[:,1], orbit[:,0])
    z = orbit[:,2]
    
    half_width = int(width/2.0)
    dt = (tlist[-1]-tlist[0])/(len(tlist)-1)

    tfreq = []
    freqR_list = []
    freqphi_list = []
    freqz_list = []

    for idx_analyze in idx_analyze_list:
        t_ = tlist[idx_analyze-half_width:idx_analyze+half_width]
        R_ = R[idx_analyze-half_width:idx_analyze+half_width]
        phi_ = phi[idx_analyze-half_width:idx_analyze+half_width]
        z_ = z[idx_analyze-half_width:idx_analyze+half_width]

        # do FT
        freq = np.fft.rfftfreq(len(t_), dt)
        ftR = np.fft.rfft(R_)
        ftphi = np.fft.rfft(phi_)
        ftz = np.fft.rfft(z_)

        freqR = freq[np.argsort(np.abs(ftR))[-2]]
        freqphi = freq[np.argsort(np.abs(ftphi))[-2]]
        freqz = freq[np.argsort(np.abs(ftz))[-2]]

        tfreq.append(tlist[idx_analyze])
        freqR_list.append(freqR)
        freqphi_list.append(freqphi)
        freqz_list.append(freqz)
    
    ans = np.transpose([tfreq, freqR_list, freqphi_list, freqz_list])

    return ans

def loop_freq(pos, tlist, idx_list, idx_analyze_list, width):
    N = pos.shape[1]
    ans = []

    half_width = int(width/2.0)
    assert idx_analyze_list[0] >= half_width
    assert len(tlist) - idx_analyze_list[-1] >= half_width
    
    R = np.linalg.norm(pos[:,:,:2], axis=2)
    phi = np.arctan2(pos[:,:,1], pos[:,:,0])
    z = pos[:,:,2]

    dt = (tlist[-1]-tlist[0])/(len(tlist)-1)

    freqR_list = []
    freqphi_list = []
    freqz_list = []

    for idx_analyze in idx_analyze_list:
        R_ = R[idx_analyze-half_width:idx_analyze+half_width,:]
        phi_ = phi[idx_analyze-half_width:idx_analyze+half_width,:]
        z_ = z[idx_analyze-half_width:idx_analyze+half_width,:]

        freq = np.fft.rfftfreq(len(R_), dt)
        ftR = np.fft.rfft(R_, axis=0)
        ftphi = np.fft.rfft(phi_, axis=0)
        ftz = np.fft.rfft(z_, axis=0)

        freqR = freq[np.argsort(np.abs(ftR), axis=0)[-2,:]]
        freqphi = freq[np.argsort(np.abs(ftphi), axis=0)[-2,:]]
        freqz = freq[np.argsort(np.abs(ftz), axis=0)[-2,:]]

        freqR_list.append(freqR)
        freqphi_list.append(freqphi)
        freqz_list.append(freqz)
    
    ans = np.array([freqR_list, freqphi_list, freqz_list])
    ans = np.swapaxes(ans, 0, 1)
    ans = np.swapaxes(ans, 1, 2)
    # order should now be (time, particle, freq)

    return np.array(ans)


# def loop_freq(pos, tlist, idx_list, idx_analyze_list, width):
#     N = pos.shape[1]
#     ans = []

#     half_width = int(width/2.0)
#     assert idx_analyze_list[0] >= half_width
#     assert len(tlist) - idx_analyze_list[-1] >= half_width
    
#     for i in range(N):
#         out = find_apoapses_compute_freq(pos[:,i,:], tlist, idx_list, idx_analyze_list, width)
#         ans.append(out)

#     return np.array(ans)

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

def _run_chunk(name, chunk_idx, prefix, phase_space_path, center, indices):
    fin = phase_space_path + name + '/phase_space_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5in = h5.File(fin, mode='r')
    
    indices = np.array(h5in['Time'])
    indices = np.arange(len(indices))

    fout = prefix + 'freq_' + name + '.' + str(chunk_idx) + '.hdf5'
    h5out = h5.File(fout, mode='w')

    tlist = np.array(h5in['Time'])

    idx_analyze_list = np.arange(100, len(tlist)-100+1, 50)
    width = 200

    # halo particles
    pos = np.array(h5in['PartType1/Coordinates'])
    ids = np.array(h5in['PartType1/ParticleIDs'])
    pos -= center
        
    ans = loop_freq(pos, tlist, indices, idx_analyze_list, width)
    h5out.create_dataset('PartType1/OmegaR', data=ans[:,:,0])
    h5out.create_dataset('PartType1/OmegaPhi', data=ans[:,:,1])
    h5out.create_dataset('PartType1/OmegaZ', data=ans[:,:,2])
    h5out.create_dataset('PartType1/ParticleIDs', data=ids)

    # load disk particles
    pos = np.array(h5in['PartType2/Coordinates'])
    ids = np.array(h5in['PartType2/ParticleIDs'])
    pos -= center
        
    ans = loop_freq(pos, tlist, indices, idx_analyze_list, width)
    h5out.create_dataset('PartType2/OmegaR', data=ans[:,:,0])
    h5out.create_dataset('PartType2/OmegaPhi', data=ans[:,:,1])
    h5out.create_dataset('PartType2/OmegaZ', data=ans[:,:,2])
    h5out.create_dataset('PartType2/ParticleIDs', data=ids)

    # load bulge particles
    pos = np.array(h5in['PartType3/Coordinates'])
    ids = np.array(h5in['PartType3/ParticleIDs'])
    pos -= center
        
    ans = loop_freq(pos, tlist, indices, idx_analyze_list, width)
    h5out.create_dataset('PartType3/OmegaR', data=ans[:,:,0])
    h5out.create_dataset('PartType3/OmegaPhi', data=ans[:,:,1])
    h5out.create_dataset('PartType3/OmegaZ', data=ans[:,:,2])
    h5out.create_dataset('PartType3/ParticleIDs', data=ids)

    # load star particles (if they exist)
    if 'PartType4' in h5in.keys():
        pos = np.array(h5in['PartType4/Coordinates'])
        ids = np.array(h5in['PartType4/ParticleIDs'])

        pos -= center
        
        ans = loop_freq(pos, tlist, indices, idx_analyze_list, width)
        h5out.create_dataset('PartType4/OmegaR', data=ans[:,:,0])
        h5out.create_dataset('PartType4/OmegaPhi', data=ans[:,:,1])
        h5out.create_dataset('PartType4/OmegaZ', data=ans[:,:,2])
        h5out.create_dataset('PartType4/ParticleIDs', data=ids)

    # for j in range(len(ans)):
    #     h5out.create_dataset('bar_metrics/'+str(ids[j]), data=ans[j])
    # h5out.create_dataset('freqs', data=ans)
    
    h5out.create_dataset('tlist', data=tlist)
    # h5out.create_dataset('id_list', data=ids)
    h5out.create_dataset('idx_list', data=indices)
    h5out.create_dataset('idx_analyze_list', data=idx_analyze_list)
    h5out.create_dataset('FrequencyTimes', data=tlist[idx_analyze_list])
    h5out.create_dataset('width', data=width)

    h5out.close()

    # bar_angle = np.mod(bar_angle_out['bar_angle'][indices], 2.*np.pi)
    # h5out.create_dataset('bar_angle', data=bar_angle)
    return 0

def run(path, name, nsnap, nproc, phase_space_path='/n/home01/abeane/starbar/plots/phase_space/data/'):
    prefix = 'data/freq_' + name +'/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # get some preliminary variables
    center, firstkey, indices = preprocess_center(name)
    
    # do standard fourier and bar angle stuff

    nchunk = len(glob.glob(phase_space_path+name+'/phase_space_'+name+'.*.hdf5'))
    print(nchunk)
    # tot_ids = []
    _ = Parallel(n_jobs=nproc) (delayed(_run_chunk)(name, i, prefix, phase_space_path, center, indices) for i in tqdm(range(nchunk)))
        
    # for i in tqdm(range(nchunk)):
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
