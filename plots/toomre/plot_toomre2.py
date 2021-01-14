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

def plot_toomre_fR(name, fout):
    fig, ax = plt.subplots(1, 1)

    fdata = 'data/' + name + '-Q2.npy'
    print(fdata)
    dat = np.load(fdata)

    R = dat[:,0]
    fR = dat[:,3]

    ax.plot(R, fR)
    ax.axhline(1.5, c = 'k', ls='dashed', alpha=0.5)
    ax.axhline(1.2, c = 'k', ls='dashed', alpha=0.5)
    ax.axhline(1.0, c = 'k', ls='dashed', alpha=0.5)

    keys = np.where(R > 1.0)[0]

    npoly = 3
    coeff = np.polyfit(R[keys], fR[keys], npoly)
    print(coeff)

    polyfit_fR = np.zeros(len(R))
    for i in range(npoly+1):
        polyfit_fR += coeff[i] * R **(npoly-i)
    ax.plot(R, polyfit_fR, c='k', ls='dashed')

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('fR')

    ax.set_ylim(0, 3)
    ax.set_xlim(0, 16)

    fig.tight_layout()
    fig.savefig(fout)

def plot_toomre_Qm1(name, fout):
    fig, ax = plt.subplots(1, 1)

    fdata = 'data/' + name + '-Q2.npy'
    print(fdata)
    dat = np.load(fdata)

    R = dat[:,0]
    Qtarget = dat[:,1]
    Qs = dat[:,5]
    Qg = dat[:,6]

    ax.plot(R, 1./Qs, label='star')
    ax.plot(R, 1./Qg, label='gas')
    ax.plot(R, 1./Qtarget, c='k', ls='dashed')
    # ax.axhline(1, c='k', ls='dashed')

    ax.legend(frameon=False)

    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('Q^-1')

    ax.set_ylim(0, 3)
    ax.set_xlim(0, 10)

    fig.tight_layout()
    fig.savefig(fout)


if __name__ == '__main__':

    fid_g1 = 'fid-disp1.0-fg0.1-lvl5'
    fid_d15_g1 = 'fid-disp1.5-fg0.1-lvl5'
    fid_g1_rcore1 = 'fid-fg0.1-Rcore1.0'
    fid_g1_rcore1_mb4 = 'fid-fg0.1-Rcore1.0-MB0.004'
    fid_g2_rcore1_mb4 = 'fid-fg0.2-Rcore1.0-MB0.004'
    fid_g1_rcore1_nomb = 'fid-fg0.1-Rcore1.0-noMB'

    name_list = [fid_g1, fid_d15_g1, fid_g1_rcore1]

    plot_toomre2(name_list, 'toomre_two_comp.pdf')
    plot_toomre_fR(fid_g1, 'toomre_fR.pdf')
    plot_toomre_Qm1(fid_g1, 'toomre_Qm1.pdf')
    
    plot_toomre_fR(fid_g1_rcore1, 'toomre_Rcore1.0_fR.pdf')
    plot_toomre_Qm1(fid_g1_rcore1, 'toomre_Rcore1.0_Qm1.pdf')
    
    plot_toomre_fR(fid_g1_rcore1_mb4, 'toomre_Rcore1.0-MB0.004_fR.pdf')
    plot_toomre_Qm1(fid_g1_rcore1_mb4, 'toomre_Rcore1.0-MB0.004_Qm1.pdf')
    plot_toomre_fR(fid_g2_rcore1_mb4, 'toomre_fg0.2-Rcore1.0-MB0.004_fR.pdf')
    plot_toomre_Qm1(fid_g2_rcore1_mb4, 'toomre_fg0.2-Rcore1.0-MB0.004_Qm1.pdf')
    
    plot_toomre_fR(fid_g1_rcore1_nomb, 'toomre_Rcore1.0-noMB_fR.pdf')
    plot_toomre_Qm1(fid_g1_rcore1_nomb, 'toomre_Rcore1.0-noMB_Qm1.pdf')

    #plot_toomre_ratio(fid_g1, 'toomre_two_comp_ratio_'+fid_g1+'.pdf')
    
