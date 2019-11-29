import matplotlib as mpl
mpl.use('Agg')

import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from scipy.stats import sigmaclip

import astropy.units as u

from matplotlib.animation import FuncAnimation

lvl = 'lvl5'

final_frame = 18
#snapbase = '/Volumes/abeaneSSD001/mwib_runs/arepo/galakos/lvl5/output'
base = '/Users/abeane/scratch/mwib_runs/arepo'
#base = '/n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo'
snapbase = base + '/galakos/'+lvl+'/output'
time_conv = 977.793 # converts time units to Myr
fout = 'movie_A2arrow_'+lvl+'.mp4'

indices = np.arange(final_frame+1)

center = np.array([0, 0, 0])
width = [30, 30, 10]

gas_vmin = 10.**(0.0)
gas_vmax = 10.**(4.0)
#star_vmin = 10.**(-0.5)
#star_vmax = 10.**(3.5)
star_vmin = 10.**(0.0)
star_vmax = 10.**(4.0)

nres = 512

res = [nres, nres, int(nres*width[2]/width[1])]
surf_xy = (width[0]/res[0])*(width[1]/res[1]) * (1000)**2
surf_xz = (width[0]/res[0])*(width[2]/res[2]) * (1000)**2
fps = 16

range_xy = [[center[0]-width[0]/2.0, center[0]+width[0]/2.0], [center[1]-width[1]/2.0, center[1]+width[1]/2.0]]
range_xz = [[center[0]-width[0]/2.0, center[0]+width[0]/2.0], [center[2]-width[2]/2.0, center[2]+width[2]/2.0]]

def animate_one_type(snap, ptype, im_xy, im_xz):

    xbool = np.logical_and(snap[ptype]['pos'][:,0] < center[0]+width[0]/2.0, snap[ptype]['pos'][:,0] > center[0]-width[0]/2.0)
    ybool = np.logical_and(snap[ptype]['pos'][:,1] < center[1]+width[1]/2.0, snap[ptype]['pos'][:,1] > center[1]-width[1]/2.0)
    zbool = np.logical_and(snap[ptype]['pos'][:,2] < center[2]+width[2]/2.0, snap[ptype]['pos'][:,2] > center[2]-width[2]/2.0)
    keys = np.where(np.logical_and(np.logical_and(xbool, ybool), zbool))[0]

    heatmap, xedges, yedges = np.histogram2d(snap[ptype]['pos'][:,0][keys], snap[ptype]['pos'][:,1][keys], 
                            bins=(res[0], res[1]), weights=snap[ptype]['mass'][keys]/surf_xy, range=range_xy)
    im_xy.set_data(heatmap.T)

    heatmap, xedges, yedges = np.histogram2d(snap[ptype]['pos'][:,0][keys], snap[ptype]['pos'][:,2][keys], 
                                bins=(res[0], res[2]), weights=snap[ptype]['mass'][keys]/surf_xz, range=range_xz)
    im_xz.set_data(heatmap.T)

    return (im_xy, im_xz)

def animate(frame, imlist, text):
    snap, time = read_snapshot(snapbase, frame, return_time=True)

    text.set_text('t='+'{0:.2f}'.format(time)+' Myr')

    for i, (im_xz, im_xy) in enumerate(np.transpose(imlist)):
        ptype = 'PartType'+str(i)
          
        # im_xy, im_xz = animate_one_type(snap, ptype, im_xy, im_xz)
        try:
            im_xy, im_xz = animate_one_type(snap, ptype, im_xy, im_xz)
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


def read_snapshot(base, number, return_time=False, subtract_center=True):
    snapfile = base+'/snapshot_'+'{0:03}'.format(number)+'.hdf5'
    f = h5.File(snapfile, mode='r')

    snap = {}
    for i, t in enumerate(['PartType0', 'PartType1', 'PartType2', 'PartType3']):
    #for i, t in enumerate(['PartType1', 'PartType2', 'PartType3']):
        if t not in list(f.keys()):
            continue
        snap[t] = {'pos': np.array(f[t]['Coordinates'])}
        try:
            # try loading in the masses from the file
            snap[t]['mass'] = np.multiply(np.array(f[t]['Masses']), 1E10)
        except:
            # read in the mass from the header
            ms = f['Header'].attrs['MassTable'][i] * 1E10
            snap[t]['mass'] = np.full(len(snap[t]['pos']), ms)

    try:
        t = 'PartType4'
        snap[t] = {'pos': np.array(f[t]['Coordinates']), 'mass': np.multiply(np.array(f[t]['Masses']), 1E10)}
    except:
        pass

    if subtract_center:
        snap = subtract_center_from_snap(snap, number)

    if return_time:
        time = f['Header'].attrs['Time']*time_conv
        f.close()
        return snap, time
    else:
        f.close()
        return snap

fig, ax_list = plt.subplots(2, 5, sharex=True, gridspec_kw={'height_ratios': [width[2], width[1]]},
                            figsize=(12,3))

# make t=0 plot
snap, time = read_snapshot(snapbase, 0, return_time=True)

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
        xbool = np.logical_and(snap[ptype]['pos'][:,0] < center[0]+width[0]/2.0, snap[ptype]['pos'][:,0] > center[0]-width[0]/2.0)
        ybool = np.logical_and(snap[ptype]['pos'][:,1] < center[1]+width[1]/2.0, snap[ptype]['pos'][:,1] > center[1]-width[1]/2.0)
        zbool = np.logical_and(snap[ptype]['pos'][:,2] < center[2]+width[2]/2.0, snap[ptype]['pos'][:,2] > center[2]-width[2]/2.0)
        keys = np.where(np.logical_and(np.logical_and(xbool, ybool), zbool))[0]
   
        heatmap, xedges, yedges = np.histogram2d(snap[ptype]['pos'][:,0][keys], snap[ptype]['pos'][:,1][keys], 
                                bins=(res[0], res[1]), weights=snap[ptype]['mass'][keys]/surf_xy, range=range_xy)
        extent = [center[0]-width[0]/2.0, center[0]+width[0]/2.0, center[1]-width[1]/2.0, center[1]+width[1]/2.0]
        im_xy = ax_xy.imshow(heatmap.T, extent=extent, origin='lower', norm=mpl.colors.LogNorm(), vmin=gas_vmin, vmax=gas_vmax)
        
        heatmap_yz, xedges, yedges = np.histogram2d(snap[ptype]['pos'][:,0][keys], snap[ptype]['pos'][:,2][keys], 
                            bins=(res[0], res[2]), weights=snap[ptype]['mass'][keys]/surf_xz, range=range_xz)
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

