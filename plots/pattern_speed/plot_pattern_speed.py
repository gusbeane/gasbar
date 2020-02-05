import matplotlib.pyplot as plt 
import numpy as np 
import pickle

def plot_pattern_speed(name_list, c_list, ls_list, fout, n=5):
    fig, ax = plt.subplots(1, 1)

    for name, c, ls in zip(name_list, c_list, ls_list):
        fdata = 'data/bar_angle_' + name + '.p'
        dat = pickle.load(open(fdata, 'rb'))

        time = dat['time']
        ba, ps = dat['poly_eval'][n]
        true_ps = dat['pattern_speed']
        t = ax.plot(time, ps, c=c, ls=ls, label=name)

        c = t[0].get_color()
        ax.scatter(time, true_ps, c=c, s=0.2)

    ax.set_xlabel('time [Myr]')
    ax.set_ylabel('pattern speed [km/s/kpc]')
    ax.legend(frameon=False)

    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':
    name_list = ['nbody-lvl5', 'nbody-lvl4', 'nbody-lvl3']
    c_list = [None, None, None]
    ls_list = [None, None, None]
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_nbody.pdf')

    name_list = ['wet-lvl5', 'wet-lvl4', 'wet-lvl3']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_wet.pdf')

    name_list = ['fid-lvl5', 'fid-lvl4', 'fid-lvl3']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_fid.pdf')

    name_list = ['nbody-lvl5', 'wet-lvl5', 'fid-lvl5']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_lvl5.pdf')
    
    name_list = ['nbody-lvl4', 'wet-lvl4', 'fid-lvl4']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_lvl4.pdf')
    
    name_list = ['nbody-lvl3', 'wet-lvl3', 'fid-lvl3']
    plot_pattern_speed(name_list, c_list, ls_list, 'pattern_speed_lvl3.pdf')

