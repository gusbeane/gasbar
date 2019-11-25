import matplotlib.pyplot as plt 
import matplotlib as mpl
import numpy as np
import glob
from tqdm import tqdm
import pickle

files = glob.glob('data/*.p')

for f in tqdm(files):
    fig, ax = plt.subplots(1, 1)

    data = pickle.load(open(f, 'rb'))

    out = data['out']   
    extent = data['extent']
    time = data['time']  

    Rbins = data['Rbins'] 
    vcirc = data['vcirc'] 
    kappa = data['kappa'] 
    name = data['name']  
    snap = data['snap']  
    m = data['m']     

    ax.imshow(out.T, origin='lower', extent=extent, aspect='auto',
        norm=mpl.colors.LogNorm(), cmap='jet')#, vmin=-1.2, vmax=5)
    
    ax.set_title('time: ' + "{0:02f}".format(time))

    # plot corotation resonance
    ax.plot(Rbins, vcirc/Rbins, c='k')
    
    # plot Linbdlad resonances
    ax.plot(Rbins, vcirc/Rbins - (kappa/m), ls='--', c='k')
    ax.plot(Rbins, vcirc/Rbins + (kappa/m), ls='--', c='k')
    
    # plot ultra-harmonic resonances
    ax.plot(Rbins, vcirc/Rbins - (kappa/(2*m)), ls='-.', c='k')
    ax.plot(Rbins, vcirc/Rbins + (kappa/(2*m)), ls='-.', c='k')

    xlim = [extent[0], extent[1]]
    ylim = [extent[2], extent[3]]

    # ax.set_yscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('pattern speed [km/s/kpc]')
    
    ax.legend(title='time [Myr]', frameon=False)
    
    fig.tight_layout()
    fig.savefig('fig/spectrogram_'+name+'_snap'+snap+'_m'+str(m)+'.pdf')