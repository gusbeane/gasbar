import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
import glob
from tqdm import tqdm
import copy

from matplotlib import rc

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/analysis/bar_prop/data/'

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def read_bar_angle(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = t['bar_angle'][:]
    t.close()

    return out

def get_center(name):
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
    else:
        center = np.array([200., 200., 200.])

    return center

def get_mass(lvl):
    ref = 7.5E3 / 1E10
    if lvl=='lvl4':
        return ref * 8
    if lvl=='lvl3':
        return ref
    if lvl=='lvl2':
        return ref / 8
    else:
        print('lvl not recognized')
        sys.exit(-1)

def rotate_pos(pos, ang):

    Rmat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang),  np.cos(ang), 0.0],
                     [0.0,         0.0,          1.0]])
    
    pos = np.swapaxes(pos, 0, 1)
    pos = np.matmul(Rmat, pos)
    pos = np.swapaxes(pos, 0, 1)
    
    return pos

def read_phase_space(name, lvl, snap_idx, phase_space_path):
    nchunk = len(glob.glob(phase_space_path+'/'+name+'-'+lvl+'/phase_space_'+name+'-'+lvl+'.*.hdf5'))

    # read in first file
    fname0 = phase_space_path+'/'+name+'-'+lvl+'/phase_space_'+name+'-'+lvl+'.0.hdf5'
    h5_f0 = h5.File(fname0, mode='r')
    #extract particle types and fields and time
    ptypes = list(h5_f0.keys())
    ptypes = [p for p in ptypes if 'PartType' in p]
    fields = h5_f0[ptypes[0]].keys()
    time = h5_f0['Time'][snap_idx]
    # close
    h5_f0.close()

    pos = []
    

    for i in tqdm(range(nchunk)):
        fname = phase_space_path+'/'+name+'-'+lvl+'/phase_space_'+name+'-'+lvl+'.'+str(i)+'.hdf5'
        h5_f = h5.File(fname, mode='r')

        pos.append(h5_f['PartType2/Coordinates'][:,snap_idx,:])
        pos.append(h5_f['PartType3/Coordinates'][:,snap_idx,:])
        if 'PartType4' in ptypes:
            pos.append(h5_f['PartType4/Coordinates'][:,snap_idx,:])
        
        h5_f.close()
    
    pos = np.concatenate(pos)

    return time, pos

def read_in_bar(name, lvl, snap_idx, in_bar_path):
    nchunk = len(glob.glob(in_bar_path+'/in_bar_'+name+'-'+lvl+'/in_bar_'+name+'-'+lvl+'.*.hdf5'))

    in_bar = []   

    for i in tqdm(range(nchunk)):
        fname = in_bar_path+'/in_bar_'+name+'-'+lvl+'/in_bar_'+name+'-'+lvl+'.'+str(i)+'.hdf5'
        h5_f = h5.File(fname, mode='r')

        in_bar.append(h5_f['in_bar'][snap_idx])
        
        h5_f.close()
    
    in_bar = np.concatenate(in_bar)

    return in_bar

def gen_heatmap(pos, mass, nres, range):

    heatmap, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=(nres, nres), range=range, weights=mass)

    dx = (range[0][1] - range[0][0]) / nres
    dy = (range[1][1] - range[1][0]) / nres
    
    heatmap /= dx * dy

    heatmap[np.isnan(heatmap)] = np.min(heatmap)
    heatmap[heatmap==0.0] = np.min(heatmap)

    return heatmap

def run(name, lvl, idx):
    phase_space_path = '/n/home01/abeane/starbar/analysis/phase_space/data'
    in_bar_path = '/n/home01/abeane/starbar/analysis/in_bar/data'
    
    center = get_center(name)
    mass = get_mass(lvl)
    nres = 256
    rng = [[-7.5, 7.5], [-7.5, 7.5]]
    extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]

    vmin = 0.005
    vmax = 1.0

    bar_angle = read_bar_angle(name, lvl)
    cm = 1/2.54

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(textwidth*cm, (1./3.)*textwidth*cm))
    
    try:
        heatmap = np.load('heatmap.npy')
        heatmap_in_bar = np.load('heatmap_in_bar.npy')
        heatmap_not_in_bar = np.load('heatmap_not_in_bar.npy')

    except:
        time, pos = read_phase_space(name, lvl, idx, phase_space_path)
        in_bar = read_in_bar(name, lvl, idx, in_bar_path)

        pos = pos - center
        pos = rotate_pos(pos, -bar_angle[idx])

        pos_in_bar = pos[in_bar]
        pos_not_in_bar = pos[np.logical_not(in_bar)]

        # return pos, mass, nres, rng
        heatmap = gen_heatmap(pos, np.full(len(pos), mass), nres, rng)
        heatmap_in_bar = gen_heatmap(pos_in_bar, np.full(len(pos_in_bar), mass), nres, rng)
        heatmap_not_in_bar = gen_heatmap(pos_not_in_bar, np.full(len(pos_not_in_bar), mass), nres, rng)

        np.save('heatmap.npy', heatmap)
        np.save('heatmap_in_bar.npy', heatmap_in_bar)
        np.save('heatmap_not_in_bar.npy', heatmap_not_in_bar)

    cmap = copy.copy(plt.get_cmap('binary'))
    # cmap.set_bad(cmap.colors[0])
    interpolation='bicubic'

    ax[0].imshow(heatmap.T, extent=extent, origin='lower', interpolation=interpolation,
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    ax[1].imshow(heatmap_in_bar.T, extent=extent, origin='lower', interpolation=interpolation,
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    ax[2].imshow(heatmap_not_in_bar.T, extent=extent, origin='lower', interpolation=interpolation,
        cmap=cmap, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))

    ax[1].plot([4.5, 6.5], [-6, -6], c='w', lw=2)
    ax[1].text(5.5, -5.5, r'$2\,\textrm{kpc}$', c='w', ha='center')


    # for im in [im0, im1, im2]:
        # im.

    ax[0].set_title('total disk')
    ax[1].set_title('barred disk')
    ax[2].set_title('unbarred disk')

    for x in ax.ravel():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
        x.set_aspect('equal')
    
    fig.tight_layout()

    fig.savefig('bar_decomp.pdf')


if __name__ == '__main__':
    name = phS2R35
    lvl = 'lvl3'
    idx = 200
    hm = run(name, lvl, idx)

