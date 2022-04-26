import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl

from matplotlib import rc
mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
bprop_path = '/n/home01/abeane/starbar/plots/bar_prop/data/'

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def read_snap(idx, name, lvl, parttype=[2, 3, 4], fields=['Coordinates', 'Masses']):
    sn = arepo.Snapshot(snap_path+name+'/'+lvl+'/output', idx, combineFiles=True, parttype=parttype,
                        fields=fields)
    return sn

def read_bar_angle(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = t['bar_angle'][:]
    t.close()

    return out

def extract_pos_mass(sn, center):
    pos = np.array([]).reshape((0, 3))
    mass = np.array([])
    for i in [2, 3, 4]:
        if sn.NumPart_Total[i] == 0:
            continue

        part = getattr(sn, 'part'+str(i))

        pos_ = part.pos.value - center
        pos = np.concatenate((pos, pos_))

        if sn.MassTable[i] > 0.0:
            mass_ = np.full(sn.NumPart_Total[i], sn.MassTable[i])
        else:
            mass_ = part.mass.value
        
        mass = np.concatenate((mass, mass_))
    
    return pos, mass

def get_center(name):
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
    else:
        center = np.array([200., 200., 200.])

    return center

def rotate_pos(pos, ang):

    Rmat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang),  np.cos(ang), 0.0],
                     [0.0,         0.0,          1.0]])
    
    pos = np.swapaxes(pos, 0, 1)
    pos = np.matmul(Rmat, pos)
    pos = np.swapaxes(pos, 0, 1)
    
    return pos

def gen_heatmap(idx, name, lvl, nres, range, bar_angle):
    sn = read_snap(idx, name, lvl)
    center = get_center(name)

    pos, mass = extract_pos_mass(sn, center)

    pos = rotate_pos(pos, -bar_angle)

    heatmap, _, _ = np.histogram2d(pos[:,0], pos[:,1], bins=(nres, nres), range=range, weights=mass)

    dx = (range[0][1] - range[0][0]) / nres
    dy = (range[1][1] - range[1][0]) / nres
    
    heatmap /= dx * dy

    return heatmap


def run():
    nres = 256
    rng = [[-7.5, 7.5], [-7.5, 7.5]]

    vmin = 0.005
    vmax = 1.0

    bar_angle_Nbody = read_bar_angle(Nbody, lvl)
    bar_angle_SMUGGLE = read_bar_angle(phS2R35, lvl)


    Nbody_idx = [500, 700, 900]
    SMUGGLE_idx = [200, 400, 600]

    name_list = [Nbody, phS2R35]

    extent = [rng[0][0], rng[0][1], rng[1][0], rng[1][1]]

    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)

    for i in range(len(Nbody_idx)):
        # plot Nbody
        ba = bar_angle_Nbody[Nbody_idx[i]]
        heatmap_N = gen_heatmap(Nbody_idx[i], Nbody, lvl, nres, rng, ba)
        print(np.min(heatmap_N), np.max(heatmap_N))
        ax[0][i].imshow(heatmap_N.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))

        # plot SMUGGLE
        ba = bar_angle_SMUGGLE[SMUGGLE_idx[i]]
        heatmap_S = gen_heatmap(SMUGGLE_idx[i], phS2R35, lvl, nres, rng, ba)
        print(np.min(heatmap_S), np.max(heatmap_S))
        ax[1][i].imshow(heatmap_S.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax))
    
    for x in ax.ravel():
        x.axes.xaxis.set_ticks([])
        x.axes.yaxis.set_ticks([])
    
    ax[0][0].set_title(r'$t=1\,\textrm{Gyr}$')
    ax[0][1].set_title(r'$t=2\,\textrm{Gyr}$')
    ax[0][2].set_title(r'$t=3\,\textrm{Gyr}$')

    ax[0][0].set_ylabel('N-body')
    ax[1][0].set_ylabel('SMUGGLE')

    fig.tight_layout()

    fig.savefig('fig1.pdf')


if __name__ == '__main__':
    run()

