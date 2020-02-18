import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import astropy.units as u

dt = 100 # Myr

def plot_SFR(name_list, c_list, ls_list, fout, n=5):
    fig, ax = plt.subplots(1, 1)

    for name, c, ls in zip(name_list, c_list, ls_list):
        fdata = 'data/SFR_' + name + '.p'
        star_birthtime, star_mini = pickle.load(open(fdata, 'rb'))

        bins = np.linspace(0, np.max(star_birthtime), dt)

        digit = np.digitize(star_birthtime, bins)

        t_list = np.array([star_birthtime[digit == i].mean() for i in range(1, len(bins))])
        SFR_list, edges = np.histogram(star_birthtime, bins=bins, weights=star_mini)

        SFR_list = SFR_list * (1/dt) * u.Msun/u.Myr
        SFR_list = SFR_list.to_value(u.Msun/u.yr)

        ax.plot(t_list, SFR_list, label=name)

    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('SFR [Msun/yr]')
    ax.legend(frameon=False)

    ax.set_yscale('log')

    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':
    name_list = ['fid-lvl4', 'fid-lvl3']#, 'nbody-lvl4', 'nbody-lvl3']
    c_list = [None, None, None]
    ls_list = [None, None, None]

    plot_SFR(name_list, c_list, ls_list, 'SFR_fid.pdf')

    name_list = ['fid-disp1.0-fg0.1-lvl5', 'fid-disp1.0-fg0.3-lvl5', 'fid-disp1.0-fg0.5-lvl5']
    plot_SFR(name_list, c_list, ls_list, 'SFR_fid-fg-lvl5.pdf')
    
    name_list = ['fid-disp1.0-fg0.1-lvl4', 'fid-disp1.0-fg0.3-lvl4', 'fid-disp1.0-fg0.5-lvl4']
    plot_SFR(name_list, c_list, ls_list, 'SFR_fid-fg-lvl4.pdf')

