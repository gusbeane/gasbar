import matplotlib as mpl
mpl.use('Agg')

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import arepo

import astropy.units as u

from matplotlib.animation import FuncAnimation

def make_movie(fout, sim, final_frame, subtract_center):

    snapbase = sim + '/output'
    time_conv = 977.793 # converts time units to Myr
    
    indices = np.arange(final_frame+1)
    
    center = np.array([0, 0, 0])
    width = [30, 30, 10]
    
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
    
        if hasattr(part.mass, 'as_unit'):
            this_mass = part.mass.as_unit(arepo.u.msol).value
        else:
            this_mass = part.mass

        heatmap, xedges, yedges = np.histogram2d(part.pos[:,0][keys], part.pos[:,1][keys], 
                                bins=(res[0], res[1]), weights=this_mass[keys]/surf_xy, range=range_xy)
        im_xy.set_data(heatmap.T)
    
        heatmap, xedges, yedges = np.histogram2d(part.pos[:,0][keys], part.pos[:,2][keys], 
                                    bins=(res[0], res[2]), weights=this_mass[keys]/surf_xz, range=range_xz)
        im_xz.set_data(heatmap.T)
    
        return (im_xy, im_xz)
    
    def animate(frame, imlist, text):
        snap, time = read_snapshot(snapbase, frame, return_time=True, subtract_center=subtract_center)
    
        text.set_text('t='+'{0:.2f}'.format(time)+' Myr')
    
        for i, (im_xz, im_xy) in enumerate(np.transpose(imlist)):
              
            im_xy, im_xz = animate_one_type(snap, i, im_xy, im_xz)
            try:
                pass
                # im_xy, im_xz = animate_one_type(snap, ptype, im_xy, im_xz)
                # pass
            except:
               pass
    
        return (imlist, text)
    
    def subtract_center_from_snap(snap, number):
    
        for i, t in enumerate(snap.keys()):
            if i==0:
                all_pos = snap[t]['pos']
                all_mass = snap[t]['mass']
            else:
                all_pos = np.concatenate((all_pos, snap[t]['pos']))
                all_mass = np.concatenate((all_mass, snap[t]['mass']))
    
        # com = np.average(all_pos, axis=0, weights=all_mass)
        com = np.median(all_pos, axis=0)
        for t in snap.keys():
            snap[t]['pos'] = np.subtract(snap[t]['pos'], com)
    
        return snap
    
    
    def read_snapshot(base, number, return_time=False, subtract_center=False):
        
        snap = arepo.Snapshot(base, number, combineFiles=True)
    
        for i, npart in enumerate(snap.NumPart_Total):
            if npart == 0:
                continue

            part = getattr(snap, 'part'+str(i))
            mass = snap.MassTable[i].as_unit(arepo.u.msol).value
            if mass > 0:
                part.mass = np.full(npart, mass)
    
        if subtract_center:
            snap = subtract_center_from_snap(snap, number)
    
        if return_time:
            time = snap.Time.as_unit(arepo.u.d).value * u.d
            time = time.to_value(u.Myr)
            return snap, time
        else:
            return snap
    
    fig, ax_list = plt.subplots(2, 5, sharex=True, gridspec_kw={'height_ratios': [width[2], width[1]]},
                                figsize=(12,3))
    
    # make t=0 plot
    snap, time = read_snapshot(snapbase, 0, return_time=True, subtract_center=subtract_center)
    # return snap

    im_list = np.zeros(np.shape(ax_list)).tolist()
    for i in range(5):
        ax_yz, ax_xy = ax_list.T[i]
        ptype = 'PartType'+str(i)
    
        ax_yz.set_title(ptype)
    
        ax_xy.set_xlim((center[0]-width[0]/2.0, center[0]+width[0]/2.0))
        ax_xy.set_ylim((center[1]-width[1]/2.0, center[1]+width[1]/2.0))
        ax_yz.set_ylim((center[2]-width[2]/2.0, center[2]+width[2]/2.0))
    
        ax_xy.set_xlabel('x [kpc]')
        ax_xy.set_ylabel('y [kpc]')
        ax_yz.set_ylabel('z [kpc]')
    
        try:
            xbool = np.logical_and(part.pos[:,0] < center[0]+width[0]/2.0, part.pos[:,0] > center[0]-width[0]/2.0)
            ybool = np.logical_and(part.pos[:,1] < center[1]+width[1]/2.0, part.pos[:,1] > center[1]-width[1]/2.0)
            zbool = np.logical_and(part.pos[:,2] < center[2]+width[2]/2.0, part.pos[:,2] > center[2]-width[2]/2.0)
            keys = np.where(np.logical_and(np.logical_and(xbool, ybool), zbool))[0]
       
            heatmap, xedges, yedges = np.histogram2d(part.pos[:,0][keys], part.pos[:,1][keys], 
                                    bins=(res[0], res[1]), weights=part.mass[keys]/surf_xy, range=range_xy)
            extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[1]-width[1]/2.0, center[1]+width[1]/2.0]
            im_xy = ax_xy.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)
            
            print('median surf den:', np.median(heatmap))

            heatmap_yz, xedges, yedges = np.histogram2d(part.pos[:,0][keys], part.pos[:,2][keys], 
                                bins=(res[0], res[2]), weights=part.mass[keys]/surf_xz, range=range_xz)
            extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[2]-width[2]/2.0, center[2]+width[2]/2.0]
            im_yz = ax_yz.imshow(heatmap_yz.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)
        
            im_list[0][i] = im_yz
            im_list[1][i] = im_xy
        except:
            heatmap = np.zeros((res[0], res[1]))
            extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[1]-width[1]/2.0, center[1]+width[1]/2.0]
            im_xy = ax_xy.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)
    
            heatmap_yz = np.zeros((res[0], res[2]))
            extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[2]-width[2]/2.0, center[2]+width[2]/2.0]
            im_yz = ax_yz.imshow(heatmap_yz.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)
    
            im_list[0][i] = im_yz
            im_list[1][i] = im_xy
    
    text = ax_list[1][-1].text(0.05, 0.9, 't='+'{0:.2f}'.format(time)+' Myr', transform=ax_list[1][-1].transAxes)
    
    fig.tight_layout()
    
    animation = FuncAnimation(fig, animate, tqdm(np.arange(final_frame+1)), fargs=[im_list, text], interval=1000 / fps)
    # animation = FuncAnimation(fig, animate, np.arange(final_frame+1), fargs=[im_list, text, arrow, arrow2], interval=1000 / fps)
    
    animation.save(fout)
    
if __name__ == '__main__':
    basepath = '../../runs/'

    nbody = 'fid-Nbody/'
    wet = 'fid-wet/'
    fid = 'fid/'
    
    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        path_list = [basepath + nbody + 'lvl5/']
        name_list = ['nbody-lvl5']
        final_frame_list = [300]
    else:
        lvl_list = [5, 4, 3, 2]
        path_list = [basepath + nbody + 'lvl' + str(i) + '/' for i in lvl_list]
        name_list = ['nbody-lvl' + str(i) for i in lvl_list]
        final_frame_list = [830, 380, 122, 14]

        lvl_wetlist = [5, 4, 3]
        wet_fflist = [830, 243, 70]
        for i,l in enumerate(lvl_wetlist):
            path_list.append(basepath + wet + 'lvl' + str(l) + '/')
            name_list.append('wet-lvl'+str(l))
            final_frame_list.append(wet_fflist[i])

    fout_list = ['movie_'+n+'.mp4' for n in name_list]

    subtract_center=False

    if len(sys.argv) > 1:
        i = int(sys.argv[1])
        make_movie(fout_list[i], path_list[i], final_frame_list[i], subtract_center)
    else:
        for fout, path, final_frame in zip(fout_list, path_list, final_frame_list):
            make_movie(fout, path, final_frame, subtract_center)
