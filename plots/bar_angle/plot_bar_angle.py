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
        true_ba = dat['bar_angle']
        phi = dat['phi']
        t = ax.plot(time, ba/(2.*np.pi), c=c, ls=ls, label=name)
        
        c = t[0].get_color()
        ax.scatter(time, true_ba/(2.*np.pi), c=c, s=0.2)
        ax.scatter(time, phi/(2.*np.pi), c=c, s=0.2)
        #ax.scatter(time[300:400], phi[300:400]/(2.*np.pi), c=c, s=0.2)

    ax.set_xlabel('time [Myr]')
    ax.set_ylabel('bar angle / 2pi')
    ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':
    name_list = ['nbody-lvl5', 'nbody-lvl4', 'nbody-lvl3']
    #name_list = ['nbody-lvl3']
    c_list = [None, None, None]
    ls_list = [None, None, None]

    plot_pattern_speed(name_list, c_list, ls_list, 'bar_angle_nbody.pdf')

    name_list = ['wet-lvl5', 'wet-lvl4', 'wet-lvl3']
    
    plot_pattern_speed(name_list, c_list, ls_list, 'bar_angle_wet.pdf')

