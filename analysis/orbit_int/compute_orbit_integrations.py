import numpy as np
import arepo
import sys
from tqdm import tqdm
import h5py as h5
import glob
import os
import re
import time
from copy import copy
from numba import njit, float64, int64
from numba.experimental import jitclass
from numba_kdtree import KDTree
from scipy import signal

import cProfile

NTYPES=6

bprop_path = '/n/home01/abeane/starbar/plots/bar_prop/data/'

# dummy tree
dummy_tree = KDTree(np.random.rand(100, 3))

spec = [
    ('tree_acc', float64[:,:]),
    ('ps', float64),
    ('omega', float64[:]),
    ('tree', dummy_tree._numba_type_)
]

@jitclass(spec)
class grav_tree(object):
    def __init__(self, tree, tree_acc, ps):
        self.tree_acc = tree_acc
        self.ps = ps
        
#         self.tree = KDtree(tree_pos)
        self.tree = tree
        self.omega = np.array([0., 0., ps])
    
    def calc_grav(self, pos, vel):
        ans = self.tree.query(pos)
        key = ans[1]
    
        out = self.tree_acc[key[0]]

        coriolis = -2. * np.cross(self.omega, vel)
        centrifugal = - np.cross(self.omega, np.cross(self.omega, pos))
        out += coriolis + centrifugal
    
        return out

    def calc_grav_N(self, pos, vel):
        ans = self.tree.query(pos)
        key = ans[1]
    
        N = pos.shape[0]
    
        out = np.zeros((N, 3))
        for i in range(len(pos)):
            out[i] = self.tree_acc[key[i]]
    
            coriolis = -2. * np.cross(self.omega, vel[i])
            centrifugal = - np.cross(self.omega, np.cross(self.omega, pos[i]))
            out[i] += coriolis + centrifugal
    
        return out

def read_snap(path, idx, parttype=[0], fields=['Coordinates', 'Masses', 'Acceleration', 'Velocities', 'ParticleIDs']):
    
    fname = path + '/output'
    
    return arepo.Snapshot(fname, idx, parttype=parttype, fields=fields, combineFiles=True)

def read_bar_angle(name, lvl):
    t = h5.File(bprop_path + 'bar_prop_' + name + '-' + lvl + '.hdf5', mode='r')
    out = t['bar_angle'][:]
    tlist = t['tlist'][:]
    t.close()
    
    fixed = fix_bar_angle(out)
    ps = np.gradient(fixed, tlist)

    return out, ps

def rotate_pos(pos, ang):

    Rmat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                     [np.sin(ang),  np.cos(ang), 0.0],
                     [0.0,         0.0,          1.0]])
    
    pos = np.swapaxes(pos, 0, 1)
    pos = np.matmul(Rmat, pos)
    pos = np.swapaxes(pos, 0, 1)
    
    return pos

def fix_bar_angle(bar_angle):
    out = np.zeros(len(bar_angle))
    out[0] = bar_angle[0]

    for i in range(1, len(bar_angle)):
        dphi = bar_angle[i] - bar_angle[i-1]
        if dphi < -np.pi:
            dphi += 2.*np.pi
        if dphi > np.pi:
            dphi -= 2. * np.pi
        
        out[i] = out[i-1] + dphi
    
    return out

def _initialize_gravity(name, lvl, snap_idx, ps, center, bangle, basepath='../../runs/'):
    global tree
    sn = read_snap(basepath+name+'/'+lvl, snap_idx, parttype=None)
    
    pos = []
    acc = []

    for pt in range(NTYPES):
        if sn.NumPart_Total[pt] > 0:
            part = getattr(sn, 'part'+str(pt))
            pos.append(part.pos.value)
            acc.append(part.acce)
    
    pos = np.concatenate(pos)
    acc = np.concatenate(acc)

    pos = pos - center
    pos = rotate_pos(pos, -bangle)

    acc = rotate_pos(acc, -bangle)

    tree = KDTree(pos)
    grav = grav_tree(tree, acc, ps)

    return grav

@njit
def integrate_orbit(dt, tmax, pos0, vel0, grav):
  
    continue_integrating = True
    
    posi = pos0
    veli = vel0
    
    t = 0.0

    pos = np.zeros((int(tmax/dt)+2, 3)).astype(np.float64)
    vel = np.zeros((int(tmax/dt)+2, 3)).astype(np.float64)
    tlist = np.zeros(int(tmax/dt)+2).astype(np.float64)

    for k in range(3):
        pos[0][k] = pos0[k]
        vel[0][k] = vel0[k]
    
    i = 0

    while continue_integrating:
        t = t + dt

        k1 = grav.calc_grav(posi, veli)[0]
        vel1 = veli + k1 * dt/2.0
        pos1 = posi + (dt/2.0)*((veli+vel1)/2.0)
        
        k2 = grav.calc_grav(pos1, vel1)[0]
        vel2 = veli + k2*dt/2.0
        pos2 = posi + (dt/2.0)*((veli+vel2)/2.0)
        
        k3 = grav.calc_grav(pos2, vel2)[0]
        vel3 = veli + k3*dt
        pos3 = posi + dt * ((veli+vel3)/2.0)
        
        k4 = grav.calc_grav(pos3, vel3)[0]
        
        velip1 = veli + (dt/6.0) * (k1 + 2.*k2 + 2.*k3 + k4)
        posip1 = posi + (dt/6.0) * (veli + 2.*vel1 + 2.*vel2 + vel3)
    
        # now determine if we need to continue integrating
        if t > tmax:
            continue_integrating = False

        # update i
        i = i+1
        for k in range(3):
            pos[i][k] = posip1[k]
            vel[i][k] = velip1[k]
        tlist[i]=t

        posi = np.copy(posip1)
        veli = np.copy(velip1)
#         acci = np.copy(accip1)

    # convert back to inertial reference frame
    ang = grav.ps * tlist
    omega = np.array([0., 0., grav.ps])
    pos_inertial = np.copy(pos)
    vel_inertial = np.copy(vel)
    
    pos_inertial[:,0] = pos[:,0] * np.cos(ang) - pos[:,1] * np.sin(ang)
    pos_inertial[:,1] = pos[:,0] * np.sin(ang) + pos[:,1] * np.cos(ang)
    
    vel_inertial[:,0] = vel[:,0] * np.cos(ang) - vel[:,1] * np.sin(ang)
    vel_inertial[:,1] = vel[:,0] * np.sin(ang) + vel[:,1] * np.cos(ang)
    vel_inertial[i] = vel_inertial[i] + np.cross(pos_inertial[i], omega)
        
    return pos_inertial, vel_inertial, tlist

def compute_phi_freq(dt, tmax, pos0, vel0, grav):
    x, v, t = integrate_orbit(dt, tmax, pos0, vel0, grav)
    phi = np.arctan2(x[:,1], x[:,0])
    R = np.linalg.norm(x[:,:2], axis=1)
    
    phi_fixed = fix_bar_angle(phi)
    
    Omega_phi = (phi_fixed[-1]-phi_fixed[0])/(t[-1]-t[0])
    
    try:
        idx, _ = signal.find_peaks(R)
        Omega_R = 2.*np.pi*(len(idx)-1) /(t[idx[-1]]-t[idx[0]])
    except:
        Omega_R = np.nan

    try:
        idx, _ = signal.find_peaks(x[:,2])
        Omega_z = np.pi * (len(idx)-1) / (t[idx[-1]]-t[idx[0]])
    except:
        Omega_z = np.nan

#     Omega_phi, Omega_R = None, None
    
    return Omega_phi, Omega_R, Omega_z

def get_halo_pos(name, lvl, snap_idx, ps, center, bangle, basepath='../../runs/'):
    sn = read_snap(basepath+name+'/'+lvl, snap_idx, parttype=1)

    pos = sn.part1.pos.value
    vel = sn.part1.vel.value
    omega = np.array([0., 0., ps])

    pos = pos - center
    pos = rotate_pos(pos, -bangle)
    vel = rotate_pos(vel, -bangle)

    vel = vel - np.cross(omega, pos)

    #sort by id
    key = np.argsort(sn.part1.id)
    pos = pos[key]
    vel = vel[key]

    return pos, vel

def get_center(name):
    if 'Nbody' in name:
        center = np.array([0., 0., 0.])
    else:
        center = np.array([200., 200., 200.])

    return center

def run(name, lvl, snap_idx, chunk_idx, nchunk, basepath='../../runs/', data_dir='data/', tmp_dir='/tmp/'):
    dt = 0.002
    tmax = 20.

    prefix = data_dir + name + '-' + lvl + '/'
    if not os.path.isdir(prefix):
        os.mkdir(prefix)

    # Read in bar angle and ps.
    bangle, ps = read_bar_angle(name, lvl)
    bangle = bangle[snap_idx]
    ps = ps[snap_idx]

    center = get_center(name)

    # Initialize gravity.
    grav = _initialize_gravity(name, lvl, snap_idx, ps, center, bangle, basepath=basepath)

    # Get halo pos and vel.
    pos_halo, vel_halo = get_halo_pos(name, lvl, snap_idx, ps, center, bangle)

    key = np.arange(len(pos_halo))
    key_split = np.array_split(key, nchunk)

    pos_to_int = pos_halo[key_split[chunk_idx]]
    vel_to_int = vel_halo[key_split[chunk_idx]]

    print('got here')
    freq = [compute_phi_freq(dt, tmax, pos_to_int[i], vel_to_int[i], grav) for i in tqdm(range(len(pos_to_int)))]
    freq = np.array(freq)

    np.save(prefix+'freqs_'+name+'-'+lvl+'.'+str(chunk_idx)+'.npy', freq)

    return freq


if __name__ == '__main__':
    name = sys.argv[1]
    lvl = sys.argv[2]
    snap_idx = int(sys.argv[3])
    chunk_idx = int(sys.argv[4])

    nchunk = 1024

    freq = run(name, lvl, snap_idx, chunk_idx, nchunk)
