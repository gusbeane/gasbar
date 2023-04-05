import numpy as np
import matplotlib.pyplot as plt
import arepo
import h5py as h5
import matplotlib as mpl
from matplotlib import rc
import glob
from tqdm import tqdm
from scipy.signal import savgol_filter

mpl.use('Agg')

rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 8})
rc('text', usetex=True)
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

basepath = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/'

snap_path = '/n/holystore01/LABS/hernquist_lab/Users/abeane/starbar_runs/runs/'
fourier_path = basepath + '/analysis/fourier_component/data/'
bprop_path = basepath + '/analysis/bar_prop/data/'
torque_path = basepath + '/analysis/torques/data/'

columnwidth = 244.0 * 0.035145980349999517 # convert to cm
textwidth = 508.0 * 0.035145980349999517 # convert to cm

tb_c = ['#4e79a7', '#f28e2b', '#e15759', '#76b7b2', '#59a14f',
        '#edc948', '#b07aa1', '#ff9da7', '#9c755f', '#bab0ac']

# names
Nbody = 'Nbody'
phS2R35 = 'phantom-vacuum-Sg20-Rc3.5'

lvl = 'lvl3'

def fix_bar_angle(bar_angle):
    out = np.zeros(len(bar_angle))
    out[0] = bar_angle[0]

    for i in range(1, len(bar_angle)):
        dphi = bar_angle[i] - bar_angle[i-1]
        if dphi < -np.pi:
            dphi += 2.*np.pi
        
        out[i] = out[i-1] + dphi
    
    return out

def read_bar_prop(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = {}
    for key in t['bar_prop'].keys():
        out[key] = t['bar_prop'][key][:]
    
    out['bar_angle'] = fix_bar_angle(t['bar_angle'][:])
    t.close()


    # print(out)
    # print(len(out['bar_angle']))
    # print(len(out['tlist']))

    out['pattern_speed'] = np.gradient(out['bar_angle'], out['tlist'])


    return out

def read_torque(name, lvl):
    base = torque_path + 'torques_' + name + '-' + lvl + '/torques_' + name + '-' + lvl + '.'

    # nfiles = len(glob.glob(base + '*.hdf5'))
    nfiles = 1200

    tz_halo = []
    tz_not_bar = []
    tz_gas = []

    tlist = []

    for i in tqdm(range(nfiles)):
        fname = base + str(i) + '.hdf5'
        t = h5.File(fname, mode='r')

        torque_halo = t['total_torques'].attrs['halo']
        tz_halo.append(torque_halo[2])

        torque_not_bar = t['total_torques'].attrs['not_bar']
        tz_not_bar.append(torque_not_bar[2])
    
        if 'gas' in t['total_torques'].attrs.keys():
            torque_gas = t['total_torques'].attrs['gas']
            tz_gas.append(torque_gas[2])
        
        tlist.append(t['parameters'].attrs['Time'])
        
        t.close()
    
    tz_halo = np.array(tz_halo)
    tz_not_bar = np.array(tz_not_bar)
    tz_gas = np.array(tz_gas)
    tlist = np.array(tlist)

    return tlist, tz_halo, tz_not_bar, tz_gas

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def run():
    bar_prop_Nbody = read_bar_prop(Nbody, lvl)
    bar_prop_SMUGGLE = read_bar_prop(phS2R35, lvl)

    cm = 1/2.54
    fig, ax = plt.subplots(1, 1, sharex=True, figsize=(columnwidth*cm, 8*cm))

    # Third panel, torques.
    tlist, tz_halo, tz_not_bar, _= read_torque(Nbody, lvl)
    tlist_g, tz_halo_g, tz_not_bar_g, tz_gas_g = read_torque(phS2R35, lvl)

    tz_halo = savgol_filter(tz_halo, 81, 3)
    tz_halo_g = savgol_filter(tz_halo_g, 81, 3)
    tz_gas_g = savgol_filter(tz_gas_g, 81, 3)
    
    tz_not_bar = savgol_filter(tz_not_bar, 81, 3)
    tz_not_bar_g = savgol_filter(tz_not_bar_g, 81, 3)

    # ax.plot(tlist-tlist[300], -tz_halo, c=tb_c[0])
    # ax.plot(tlist_g, -tz_halo_g, c=tb_c[1])
    # ax.plot(tlist_g, -tz_gas_g, c=tb_c[1], ls='dashed')
    
    ax.plot(tlist-tlist[300], -tz_not_bar, c=tb_c[0], label=r'$N$-body')#, ls='dotted')
    ax.plot(tlist_g, -tz_not_bar_g, c=tb_c[1], label=r'SMUGGLE')#, ls='dotted')
    
    print('disk Nbody', np.mean(-tz_not_bar[300+400:300+800]))
    print('halo Nbody', np.mean(-tz_halo[300+400:300+800]))
    print('disk SMUGGLE', np.mean(-tz_not_bar_g[400:800]))
    print('halo SMUGGLE', np.mean(-tz_halo_g[400:800]))
    print('gas SMUGGLE', np.mean(-tz_gas_g[400:800]))

    ax.axhline(0, c='k')

    ax.set(xlim=(0, 4), ylim=(-100, 100), ylabel=r'$\tau_{\text{on bar}}\,[\,10^{10}\,M_{\odot}\,(\text{km}/\text{s})^2\,]$')
    ax.set_xlabel(r'$t\,[\,\text{Gyr}\,]$')

    # custom_lines = [mpl.lines.Line2D([0], [0], color='k'),
    #                 mpl.lines.Line2D([0], [0], color='k', ls='dashed'),
    #                 mpl.lines.Line2D([0], [0], color='k', ls='dotted')]
    # ax.legend(custom_lines, ['by halo', 'by gas', 'by disk'], frameon=False)
    
    ax.legend(frameon=False)
    

    fig.tight_layout()

    fig.savefig('torque_disk.pdf')

    # print average torqu
    
    # for t, tz, tzg in zip([tlist, tlist_g], [tz_halo, tz_halo_g], [None, tz_gas_g]):
    #     key = np.logical_and(t > 2, t < 4)
    #     print(np.mean(-tz[key]))
    #     if tzg is not None:
    #         print(np.mean(-tzg[key]))

#     # now do talk plot

#     fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
#     ax.plot(bar_prop_Nbody['tlist'] - t300, bar_prop_Nbody['pattern_speed'], c=tb_c[0], label='N-body')
#     ax.set(ylim=(0, 60), ylabel=r'$\Omega_p\,[\,\text{km}/\text{s}/\text{kpc}\,]$')
#     ax.set(xlim=(0, 5), xlabel=r'$t\,[\,\text{Gyr}\,]$')
#     ax.legend(frameon=False)

#     fig.tight_layout()
#     fig.savefig('talk_fig2aa.pdf')

#     ax.plot(bar_prop_SMUGGLE['tlist'], bar_prop_SMUGGLE['pattern_speed'], c=tb_c[1], label='SMUGGLE')
#     ax.legend(frameon=False)

#     fig.tight_layout()
#     fig.savefig('talk_fig2ab.pdf')


#     fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))
#     ax.plot(tlist-tlist[300], -tz_halo, c=tb_c[0])
#     # ax.plot(tlist_g, -tz_halo_g, c=tb_c[1])
#     # ax.plot(tlist_g, -tz_gas_g, c=tb_c[1], ls='dashed')

#     ax.axhline(0, c='k')

#     ax.set(xlim=(0, 5), ylim=(-100, 100), ylabel=r'$\tau_{\text{on bar}}\,[\,10^{10}\,M_{\odot}\,(\text{km}/\text{s})^2\,]$')
#     ax.set_xlabel(r'$t\,[\,\text{Gyr}\,]$')

    

#     fig.tight_layout()
#     fig.savefig('talk_fig2c.pdf')

#     custom_lines = [mpl.lines.Line2D([0], [0], color='k'),
#                     mpl.lines.Line2D([0], [0], color='k', ls='dashed')]
#     ax.legend(custom_lines, ['by halo', 'by gas'], frameon=False)

#     ax.plot(tlist_g, -tz_halo_g, c=tb_c[1])
#     ax.plot(tlist_g, -tz_gas_g, c=tb_c[1], ls='dashed')

#     fig.tight_layout()
#     fig.savefig('talk_fig2cb.pdf')

if __name__ == '__main__':
    run()

