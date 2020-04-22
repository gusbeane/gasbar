import matplotlib.pyplot as plt 
import numpy as np 
import pickle

def plot_pattern_speed(name_list, c_list, ls_list, fout, n=5, vline=None):
    fig, ax = plt.subplots(1, 1)

    for name, c, ls in zip(name_list, c_list, ls_list):
        fdata = 'data/bar_angle_' + name + '.p'
        dat = pickle.load(open(fdata, 'rb'))

        time = dat['time']
        ba, ps = dat['poly_eval'][n]
        true_ps = dat['pattern_speed']
        t = ax.plot(time-time[0], ps, c=c, ls=ls, label=name)

        c = t[0].get_color()
        ax.scatter(time-time[0], true_ps, c=c, s=0.2)

    if vline is not None:
        ax.axvline(vline, c='k', alpha=0.5, ls='dashed')

    ax.set_xlabel('time [Myr]')
    ax.set_ylabel('pattern speed [km/s/kpc]')
    ax.legend(frameon=False)

    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':
    nbody = 'nbody'
    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_wet_fixed = 'fid-wet-disp1.0-fixedDisk'
    fid_g1_fixed1kpc = 'fid-disp1.0-fixedDisk-core1kpc'
    fid_g1_fixed2kpc = 'fid-disp1.0-fixedDisk-core2kpc'
    fid_g1_fixed3kpc = 'fid-disp1.0-fixedDisk-core3kpc'
    fid_g1_fixed4kpc = 'fid-disp1.0-fixedDisk-core4kpc' 
    fid_g1_fixed5kpc = 'fid-disp1.0-fixedDisk-core5kpc' 
    fid_g1_fixed6kpc = 'fid-disp1.0-fixedDisk-core6kpc' 
    fid_g1_dS_out_delay = 'fid-disp1.0-fg0.1-diskAGB-outer-delay1.0'

    name_list = ['nbody-lvl5', 'nbody-lvl4', 'nbody-lvl3']
    c_list = [None, None, None, None, None, None]
    ls_list = [None, None, None, None, None, None]
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_nbody.pdf')

    # name_list = ['wet-lvl5', 'wet-lvl4', 'wet-lvl3']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_wet.pdf')

    # name_list = ['fid-lvl5', 'fid-lvl4', 'fid-lvl3']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_fid.pdf')

    # name_list = ['nbody-lvl5', 'wet-lvl5', 'fid-lvl5']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_lvl5.pdf')
    
    # name_list = ['nbody-lvl4', 'wet-lvl4', 'fid-lvl4']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_lvl4.pdf')
    
    # name_list = ['nbody-lvl3', 'wet-lvl3', 'fid-lvl3']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_lvl3.pdf')

    # name_list = ['nbody-lvl5', 'fid-disp1.0-fixedDisk-lvl5']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_fixedDisk_lvl5.pdf')
    
    # name_list = ['nbody-lvl4', 'fid-disp1.0-fixedDisk-lvl4']
    # plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_fixedDisk_lvl4.pdf')

    name_list = [nbody+'-lvl5', fid_wet_fixed+'-lvl5', fid_g1_fixed1kpc+'-lvl5', 
                 fid_g1_fixed2kpc+'-lvl5', fid_g1_fixed3kpc+'-lvl5', fid_g1_fixed4kpc+'-lvl5']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_fixedDisk_lvl5.pdf', vline=1.5352*1000)

    name_list = [nbody+'-lvl4', fid_wet_fixed+'-lvl4', #fid_g1_fixed1kpc+'-lvl4', 
                 fid_g1_fixed2kpc+'-lvl4', 
                 #fid_g1_fixed3kpc+'-lvl4', 
                 fid_g1_fixed4kpc+'-lvl4',
                 fid_g1_fixed5kpc+'-lvl4',
                 fid_g1_fixed6kpc+'-lvl4']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_fixedDisk_lvl4.pdf', vline=1.5352*1000)

    name_list = [nbody+'-lvl5', fid_g1+'-lvl5', fid_g1_dS_out_delay+'-lvl5']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_AGBdelay_lvl5.pdf')
    
    name_list = [nbody+'-lvl4', fid_g1+'-lvl4', fid_g1_dS_out_delay+'-lvl4']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_AGBdelay_lvl4.pdf')
    
    name_list = [nbody+'-lvl3', fid_g1_dS_out_delay+'-lvl3']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_AGBdelay_lvl3.pdf')

