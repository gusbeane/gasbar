import matplotlib.pyplot as plt 
import numpy as np 

def plot_toomre2(name_list, fout):
    fig, ax = plt.subplots(1, 1)

    for i, name in enumerate(name_list):

        fdata = 'data/' + name + '-Q2.npy'
        dat = np.load(fdata)

        R = dat[:,0]
        Q = dat[:,1]
        Q2 = dat[:,2]

        if i==0:
            ax.plot(R, Q, label='star', c='k', ls='dashed')
        ax.plot(R, Q2, label=name)
    
    ax.axhline(1, c='k')

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Q')
    ax.legend(frameon=False)

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    fig.tight_layout()
    fig.savefig(fout)

def plot_toomre_ratio(name, fout):
    fig, ax = plt.subplots(1, 1)

    fdata = 'data/' + name + '-Q2.npy'
    dat = np.load(fdata)

    R = dat[:,0]
    Q = dat[:,1]
    Q2_1 = dat[:,2]
    Q2_2 = dat[:,3]
    Q2_3 = dat[:,4]
    Q2_4 = dat[:,5]

    ax.plot(R, Q2_1/Q, label='1e1')
    ax.plot(R, Q2_2/Q, label='1e2')
    ax.plot(R, Q2_3/Q, label='1e3')
    ax.plot(R, Q2_4/Q, label='1e4')
    ax.axhline(1, c='k')

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Q2/Q')
    ax.legend(title='T [K]', frameon=False)



    fig.tight_layout()
    fig.savefig(fout)

if __name__ == '__main__':

    fid_g1 = 'fid-disp1.0-fg0.1-lvl5'
    fid_d15_g1 = 'fid-disp1.5-fg0.1-lvl5'
   
    name_list = [fid_g1, fid_d15_g1]

    plot_toomre2(name_list, 'toomre_two_comp.pdf')
    #plot_toomre_ratio(fid_g1, 'toomre_two_comp_ratio_'+fid_g1+'.pdf')
    
