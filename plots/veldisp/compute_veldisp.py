import numpy as np
import arepo
import sys
from tqdm import tqdm
import astropy.units as u
import pickle

Rmin = 8.0
Rmax = 8.2

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average(np.square(np.subtract(values, average)), weights=weights)
    return (average, np.sqrt(variance))

def compute_veldisp(path, snapnum):
    # try loading snapshot
    sn = arepo.Snapshot(path+'output/', snapnum, combineFiles=True)

    time = sn.Time.as_unit(arepo.u.d).value * u.d
    time = time.to_value(u.Myr)
    
    firstpart = True
    for i in [2, 3, 4]:
        numpart = sn.NumPart_Total[i]
        if numpart == 0:
            continue
        
        part = getattr(sn, 'part'+str(i))

        pos_in_kpc = part.pos.as_unit(arepo.u.kpc).value
        vel_in_kms = part.vel.as_unit(arepo.u.kms).value

        x, y, z = pos_in_kpc[:,0], pos_in_kpc[:,1], pos_in_kpc[:,2]
        vx, vy, vz = vel_in_kms[:,0], vel_in_kms[:,1], vel_in_kms[:,2]

        R = np.sqrt(np.add(np.square(x), np.square(y)))
        phi = np.arctan2(y, x)

        sphi = np.sin(phi)
        cphi = np.cos(phi)

        vR = np.add(np.multiply(vx, cphi), np.multiply(vy, sphi))
        vphi = np.subtract(np.multiply(vy, cphi), np.multiply(vx, sphi))

        this_pos = np.transpose([R, phi, z])
        this_vel = np.transpose([vR, vphi, vz])
        if sn.MassTable[i].value == 0:
            this_mass = part.mass.as_unit(arepo.u.msol).value
        else:
            this_mass = np.full(numpart, sn.MassTable[i].as_unit(arepo.u.msol).value)

        keys = np.logical_and(R > Rmin, R < Rmax)

        if firstpart:
            pos = np.copy(this_pos[keys])
            vel = np.copy(this_vel[keys])
            mass = np.copy(this_mass[keys])
            firstpart=False
        else:
            pos = np.concatenate((pos, this_pos[keys]))
            vel = np.concatenate((vel, this_vel[keys]))
            mass = np.concatenate((mass, this_mass[keys]))

    avevR, stdvR = weighted_avg_and_std(vel[:,0], mass)
    avevphi, stdvphi = weighted_avg_and_std(vel[:,1], mass)
    avevz, stdvz = weighted_avg_and_std(vel[:,2], mass)

    return time, stdvR, stdvphi, stdvz

def master_veldisp(path, name, final_frame, output_dir='data/'):
    snapnum_list = np.arange(final_frame+1)
    time_list = []
    stdvR_list = []
    stdvphi_list = []
    stdvz_list = []
    for snapnum in tqdm(snapnum_list):
        time, stdvR, stdvphi, stdvz = compute_veldisp(path, snapnum)
        time_list.append(time)
        stdvR_list.append(stdvR)
        stdvphi_list.append(stdvphi)
        stdvz_list.append(stdvz)

    out = np.transpose([time_list, stdvR_list, stdvphi_list, stdvz_list])
    pickle.dump(out, open(output_dir+'veldisp_'+name+'.p', 'wb'))


if __name__ == '__main__':

    basepath = '../../runs/'

    nbody = 'fid-Nbody/'
    wet = 'fid-wet/'
    fid = 'fid/'
    
    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        path_list = [basepath + nbody + 'lvl5/']
        name_list = ['nbody-lvl5']
        final_frame_list = [125]
    else:
        lvl_list = [5, 4, 3, 2]
        path_list = [basepath + nbody + 'lvl' + str(i) + '/' for i in lvl_list]
        name_list = ['nbody-lvl' + str(i) for i in lvl_list]
        final_frame_list = [620, 620, 620, 620]
    
    for path, name, final_frame in zip(tqdm(path_list), name_list, final_frame_list):
        master_veldisp(path, name, final_frame)
    