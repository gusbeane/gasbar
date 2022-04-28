import matplotlib.pyplot as plt 
import numpy as np 
import pickle

def plot_SFR_profile(name_list, c_list, ls_list, fout, n=5):
    fig, ax = plt.subplots(1, 1)

    for name, c, ls in zip(name_list, c_list, ls_list):
        fdata = 'data/SFR_profile_' + name + '.p'
        R, sfr = pickle.load(open(fdata, 'rb'))
        ax.plot(R, sfr, c=c, ls=ls, label=name) #  1E6 makes it /pc^2

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('SFR density [Msun/yr/pc^2]')
    ax.legend(frameon=False)
    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':
    
    fid_g1 = 'fid-disp1.0-fg0.1'
    # fid_wet_fixed = 'fid-wet-disp1.0-fixedDisk'
    # fid_g1_fixed1kpc = 'fid-disp1.0-fixedDisk-core1kpc'
    # fid_g1_fixed2kpc = 'fid-disp1.0-fixedDisk-core2kpc'
    # fid_g1_fixed3kpc = 'fid-disp1.0-fixedDisk-core3kpc'
    # fid_g1_fixed4kpc = 'fid-disp1.0-fixedDisk-core4kpc' 
    # fid_g1_fixed5kpc = 'fid-disp1.0-fixedDisk-core5kpc' 
    # fid_g1_fixed6kpc = 'fid-disp1.0-fixedDisk-core6kpc' 

    name_list = [fid_g1+'-lvl4'+'_snap10', fid_g1+'-lvl4'+'_snap50', fid_g1+'-lvl4'+'_snap100',
                 fid_g1+'-lvl4'+'_snap150', fid_g1+'-lvl4'+'_snap200']
    c_list = [None, None, None, None, None, None]
    ls_list = [None, None, None, None, None, None]
    plot_SFR_profile(name_list, c_list, ls_list, 'SFR_fid_lvl4_firstGyr.pdf')
    
    name_list = [fid_g1+'-lvl5'+'_snap10', fid_g1+'-lvl5'+'_snap100', fid_g1+'-lvl4'+'_snap200', fid_g1+'-lvl4'+'_snap300',
                 fid_g1+'-lvl4'+'_snap500']
    plot_SFR_profile(name_list, c_list, ls_list, 'SFR_fid_lvl4.pdf')

