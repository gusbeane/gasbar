import matplotlib.pyplot as plt 
import numpy as np 
import pickle
import astropy.units as u

time_conv = 977.793

def plot_gas_fraction(pair_list, c_list, ls_list, fout, output_dir='data/', 
                      xlim=[0, None], ylim=[None, None]):
    name_list = [           p[0] + '-' + p[1] for p in pair_list]

    fig, ax = plt.subplots(1, 1)

    max_time = 0
    for name, c, ls in zip(name_list, c_list, ls_list):
        fdata = output_dir+'gas-fraction_'+name+'.p'
        dat = pickle.load(open(fdata, 'rb'))

        time = dat[:,0]
        time = time - time[0]
        gas_fraction_disk = dat[:,1]
        gas_fraction_R0 = dat[:,2]
        gas_fraction_Rcenter = dat[:,3]

        l = ax.plot(time, gas_fraction_disk, label=name)
        c = l[0].get_color()
        ax.plot(time, gas_fraction_R0, ls='dashed', c=c)
        ax.plot(time, gas_fraction_Rcenter, ls='dashdot', c=c)

        if np.max(time) > max_time:
            max_time = np.max(time)

    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('gas fraction')
    ax.legend(frameon=False)

    # add band around MW values
    # ax.axhline(0.1, c='k', alpha=0.5)
    # ax.axhline(0.2, c='k', alpha=0.5)
    ax.fill_between([0, max_time], 0.1, 0.2, color='k', alpha=0.3)


    # ax.set_yscale('log')
    ax.set_ylim(ylim)

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
    
    fid_da = 'fid-disp1.0-fg0.1-diskAcc1.0'
    fid_da_am = 'fid-disp1.0-fg0.1-diskAcc1.0-decAngMom'

    fid_g1_dS_out_delay = 'fid-disp1.0-fg0.1-diskAGB-outer-delay1.0' 

    c_list = [None, None, None, None, None]
    ls_list = [None, None, None, None, None]
    ylim = [0, None]

    #pair_list = [(fid_g1, 'lvl5'), (fid_g2, 'lvl5'), (fid_g3, 'lvl5'), (fid_g5, 'lvl5')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-lvl5.pdf', ylim=ylim)
    
    #pair_list = [(fid_g1, 'lvl4'), (fid_g2, 'lvl4'), (fid_g3, 'lvl4'), (fid_g5, 'lvl4')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-lvl4.pdf', ylim=ylim)

    #pair_list = [(fid_g1, 'lvl3'), (fid_g2, 'lvl3'), (fid_g3, 'lvl3')]#, (fid_g5, 'lvl4')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-lvl3.pdf', ylim=ylim)
    
    #pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), (fid_g1, 'lvl3')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-g1.pdf', ylim=ylim)
    
    #pair_list = [(fid_g2, 'lvl5'), (fid_g2, 'lvl4'), (fid_g2, 'lvl3')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-g2.pdf', ylim=ylim)
    
    #pair_list = [(fid_g3, 'lvl5'), (fid_g3, 'lvl4'), (fid_g3, 'lvl3')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-g3.pdf', ylim=ylim)
    
    #pair_list = [(fid_g4, 'lvl5'), (fid_g4, 'lvl4')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-g4.pdf', ylim=ylim)
    
    #pair_list = [(fid_g5, 'lvl5'), (fid_g5, 'lvl4')]
    #plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-fg-g5.pdf', ylim=ylim)
    
    pair_list = [(fid_g1, 'lvl5'), (fid_da, 'lvl5'), (fid_da_am, 'lvl5')]
    plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-da-l5.pdf', ylim=ylim)
    
    pair_list = [(fid_g1, 'lvl4'), (fid_da, 'lvl4'), (fid_da_am, 'lvl4')]
    plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-da-l4.pdf', ylim=ylim)
    
    pair_list = [(fid_g1, 'lvl5'), (fid_g1_dS_out_delay, 'lvl5')]
    plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-AGB-delay-l5.pdf', ylim=ylim)
    
    pair_list = [(fid_g1, 'lvl4'), (fid_g1_dS_out_delay, 'lvl4')]
    plot_gas_fraction(pair_list, c_list, ls_list, 'gas-fraction_fid-AGB-delay-l4.pdf', ylim=ylim)


