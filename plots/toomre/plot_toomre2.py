import matplotlib.pyplot as plt 
import numpy as np 

def plot_toomre2(name, fout):
    fig, ax = plt.subplots(1, 1)

    fdata = 'data/' + name + '-Q2.npy'
    dat = np.load(fdata)

    R = dat[:,0]
    Q = dat[:,1]
    Q2_1 = dat[:,2]
    Q2_2 = dat[:,3]
    Q2_3 = dat[:,4]
    Q2_4 = dat[:,5]

    fac = 1.0
    fac = np.sqrt(1.5)

    ax.plot(R, fac*Q, label='star', c='k', ls='dashed')
    ax.plot(R, fac*Q2_1, label='1e1')
    ax.plot(R, fac*Q2_2, label='1e2')
    ax.plot(R, fac*Q2_3, label='1e3')
    ax.plot(R, fac*Q2_4, label='1e4')
    ax.axhline(1, c='k')

    ax.set_yscale('log')

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Q2')
    ax.legend(title='T [K]', frameon=False)

    ax.set_ylim(0.06, 5)


    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':

    name = 'fid-disp1.0-fg0.1-lvl5'

    plot_toomre2(name, 'toomre_two_comp.pdf')

