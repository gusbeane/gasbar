import matplotlib as mpl
mpl.use('Agg')

import glob
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import arepo
import os

import astropy.units as u

from matplotlib.animation import FuncAnimation

def make_movie(fout, sim, nsnap):

    print('making movie for: ' + sim)

    if 'nbody' in fout:
        center = np.array([0, 0, 0])
    else:
        center = np.array([200, 200, 200])

    snapbase = sim + '/output'
    time_conv = 977.793 # converts time units to Myr
    
    indices = np.arange(nsnap)
    
    width = [100, 100, 100]
    
    gas_vmin = 10.**(0.0)
    gas_vmax = 10.**(4.0)
    #star_vmin = 10.**(-0.5)
    #star_vmax = 10.**(3.5)
    star_vmin = 10.**(0.0)
    star_vmax = 10.**(4.0)
    
    nres = 256 
    
    res = [nres, nres, int(nres*width[2]/width[1])]
    surf_xy = (width[0]/res[0])*(width[1]/res[1]) * (1000)**2
    surf_xz = (width[0]/res[0])*(width[2]/res[2]) * (1000)**2
    fps = 16
    
    range_xy = [[center[0]-width[0]/2.0, center[0]+width[0]/2.0], [center[1]-width[1]/2.0, center[1]+width[1]/2.0]]
    range_xz = [[center[0]-width[0]/2.0, center[0]+width[0]/2.0], [center[2]-width[2]/2.0, center[2]+width[2]/2.0]]
    
    def animate_one_type(snap, pidx, im_xy, im_xz):
   
        if snap.NumPart_Total[pidx] == 0:
            return (im_xy, im_xz)
        
        part = getattr(snap, 'part'+str(pidx))

        xbool = np.logical_and(part.pos[:,0] < center[0]+width[0]/2.0, part.pos[:,0] > center[0]-width[0]/2.0)
        ybool = np.logical_and(part.pos[:,1] < center[1]+width[1]/2.0, part.pos[:,1] > center[1]-width[1]/2.0)
        zbool = np.logical_and(part.pos[:,2] < center[2]+width[2]/2.0, part.pos[:,2] > center[2]-width[2]/2.0)
        keys = np.where(np.logical_and(np.logical_and(xbool, ybool), zbool))[0]
    
        masstable = snap.MassTable[pidx].as_unit(arepo.u.msol).value
        if hasattr(part, 'mass'):
            this_mass = part.mass.as_unit(arepo.u.msol).value
        else:
            this_mass = np.full(len(part.pos), masstable)

        heatmap, xedges, yedges = np.histogram2d(part.pos[:,0][keys], part.pos[:,1][keys], 
                                bins=(res[0], res[1]), weights=this_mass[keys]/surf_xy, range=range_xy)
        im_xy.set_data(heatmap.T)
    
        heatmap, xedges, yedges = np.histogram2d(part.pos[:,0][keys], part.pos[:,2][keys], 
                                    bins=(res[0], res[2]), weights=this_mass[keys]/surf_xz, range=range_xz)
        im_xz.set_data(heatmap.T)
    
        return (im_xy, im_xz)
    
    def animate(frame, imlist, text):
        snap, time = read_snapshot(snapbase, frame, return_time=True)
    
        text.set_text('t='+'{0:.2f}'.format(time)+' Myr')
    
        im_xz, im_xy = imlist

        im_xy, im_xz = animate_one_type(snap, 0, im_xy, im_xz)
    
        return (imlist, text)

    def gen_first_frame(ax_list):
        im_list = np.zeros(np.shape(ax_list)).tolist()
        
        ax_yz, ax_xy = ax_list
    
        ax_xy.set_xlim((center[0]-width[0]/2.0, center[0]+width[0]/2.0))
        ax_xy.set_ylim((center[1]-width[1]/2.0, center[1]+width[1]/2.0))
        ax_yz.set_ylim((center[2]-width[2]/2.0, center[2]+width[2]/2.0))
    
        ax_xy.set_xlabel('x [kpc]')
        ax_xy.set_ylabel('y [kpc]')
        ax_yz.set_ylabel('z [kpc]')
    
        heatmap = np.zeros((res[0], res[1]))
        extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[1]-width[1]/2.0, center[1]+width[1]/2.0]
        im_xy = ax_xy.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)

        heatmap_yz = np.zeros((res[0], res[2]))
        extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[2]-width[2]/2.0, center[2]+width[2]/2.0]
        im_yz = ax_yz.imshow(heatmap_yz.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)

        im_list[0] = im_yz
        im_list[1] = im_xy

        return im_list, ax_list
    
    
    def read_snapshot(base, number, return_time=False):
        
        snap = arepo.Snapshot(base, number, combineFiles=True)
    
        for i, npart in enumerate(snap.NumPart_Total):
            if npart == 0:
                continue

            part = getattr(snap, 'part'+str(i))
    
        if return_time:
            time = snap.Time.as_unit(arepo.u.d).value * u.d
            time = time.to_value(u.Myr)
            return snap, time
        else:
            return snap
    
    fig, ax_list = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [width[2], width[1]]},
                                figsize=(4,8))
    
    # make t=0 plot
    snap, time = read_snapshot(snapbase, 0, return_time=True)
    # return snap

    im_list, ax_list = gen_first_frame(ax_list)
    
    text = ax_list[1].text(0.05, 0.9, 't='+'{0:.2f}'.format(time)+' Myr', transform=ax_list[1].transAxes)
    
    fig.tight_layout()
    
    animation = FuncAnimation(fig, animate, tqdm(np.arange(nsnap)), fargs=[im_list, text], interval=1000 / fps)
    # animation = FuncAnimation(fig, animate, np.arange(final_frame+1), fargs=[im_list, text, arrow, arrow2], interval=1000 / fps)
    
    animation.save(fout)
    
if __name__ == '__main__':
    basepath = '../../runs/'

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_g2 = 'fid-disp1.0-fg0.2'
    fid_g3 = 'fid-disp1.0-fg0.3'
    fid_g4 = 'fid-disp1.0-fg0.4'
    fid_g5 = 'fid-disp1.0-fg0.5'
    fid_da = 'fid-disp1.0-fg0.1-diskAcc1.0'
    fid_da_dm = 'fid-disp1.0-fg0.1-diskAcc-decAngMom'

    fid_d7_g3 = 'fid-disp0.7-fg0.3'
    fid_d5_g3 = 'fid-disp0.5-fg0.3'
    fid_g3_nB = 'fid-disp1.0-fg0.3-noBulge'

    fid_g1_corona = 'fid-disp1.0-fg0.1-corona'
    fid_g1_coronaRot = 'fid-disp1.0-fg0.1-coronaRot'
    fid_g1_coronaMat = 'fid-disp1.0-fg0.1-corona-Matthew'
    fid_g1_coronaMat4 = 'fid-disp1.0-fg0.1-corona-Matthew-MHG0.004'
   
    # look to see if we are on my macbook or on the cluster
    pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), (fid_g1, 'lvl3'),
                 (fid_g2, 'lvl5'), (fid_g2, 'lvl4'), (fid_g2, 'lvl3'),
                 (fid_g3, 'lvl5'), (fid_g3, 'lvl4'), (fid_g3, 'lvl3'),
                 (fid_g4, 'lvl5'), (fid_g4, 'lvl4'),
                 (fid_g5, 'lvl5'), (fid_g5, 'lvl4'),
                 (fid_g1_corona, 'lvl5'), (fid_g1_corona, 'lvl4'),
                 (fid_g1_coronaRot, 'lvl5'), (fid_g1_coronaRot, 'lvl4'),
                 (fid_g1_coronaMat, 'lvl5'), (fid_g1_coronaMat, 'lvl4'),
                 (fid_g1_coronaMat4, 'lvl5'), (fid_g1_coronaMat4, 'lvl4')]
                 #(fid_d7_g3, 'lvl5'), (fid_d7_g3, 'lvl4'),
                 #(fid_d5_g3, 'lvl5'), (fid_d5_g3, 'lvl4'),
                 #(fid_g3_nB, 'lvl5'), (fid_g3_nB, 'lvl4'),
                 #(fid_da, 'lvl5'), (fid_da, 'lvl4'),
                 #(fid_da_dm, 'lvl5'), (fid_da_dm, 'lvl4')]
    
    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
                                            
    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]
    fout_list = ['movies/movie_'+n+'.mp4' for n in name_list]

    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        if os.path.exists(fout_list[i]):
            sys.exit(0) 
        make_movie(fout_list[i], path_list[i], nsnap_list[i])
    else:
        for fout, path, nsnap in zip(fout_list, path_list, nsnap_list):
            if os.path.exists(fout):
                continue
            make_movie(fout, path, nsnap)

