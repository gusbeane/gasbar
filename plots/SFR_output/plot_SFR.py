import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import astropy.units as u

time_conv = 977.793

def plot_SFR(pair_list, c_list, ls_list, fout, n=5, basepath = '../../runs/'):
    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
    
    fig, ax = plt.subplots(1, 1)

    for name, path, c, ls in zip(name_list, path_list, c_list, ls_list):
        fdata = path + '/output/sfr.txt'
        dat = np.genfromtxt(fdata)

        time = dat[:,0] * time_conv
        sfr = dat[:,2] # in Msun/yr

        ax.plot(time, sfr, label=name)

    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('SFR [Msun/yr]')
    ax.legend(frameon=False)

    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':
    nbody = 'fid-Nbody'
    wet = 'fid-wet'
    fid = 'fid'
    fid_rdisk = 'fid-disp1.0-resetDisk'

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_g2 = 'fid-disp1.0-fg0.2'
    fid_g3 = 'fid-disp1.0-fg0.3'
    fid_g4 = 'fid-disp1.0-fg0.4'
    fid_g5 = 'fid-disp1.0-fg0.5'
    
    c_list = [None, None, None]
    ls_list = [None, None, None]

    pair_list = [(fid_g1, 'lvl5'), (fid_g3, 'lvl5'), (fid_g5, 'lvl5')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-lvl5.pdf')
    
    pair_list = [(fid_g1, 'lvl4'), (fid_g3, 'lvl4'), (fid_g5, 'lvl4')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-lvl4.pdf')
    
    pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-g1.pdf')
    
    pair_list = [(fid_g2, 'lvl5'), (fid_g2, 'lvl4')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-g2.pdf')
    
    pair_list = [(fid_g3, 'lvl5'), (fid_g3, 'lvl4')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-g3.pdf')
    
    pair_list = [(fid_g4, 'lvl5'), (fid_g4, 'lvl4')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-g4.pdf')
    
    pair_list = [(fid_g5, 'lvl5'), (fid_g5, 'lvl4')]
    plot_SFR(pair_list, c_list, ls_list, 'SFR_fid-fg-g5.pdf')

