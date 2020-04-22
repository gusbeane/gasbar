import numpy as np
import arepo
import sys
from tqdm import tqdm
import h5py as h5
import pickle
import re
import astropy.units as u

Rbin = 5
firstkey = 150
nmax = 10

def get_bar_angle(phi, firstkey):
    out = np.zeros(len(phi))

    # set the first bar angle
    first_bar_angle = phi[firstkey]/2.0
    out[firstkey] = first_bar_angle
    
    # set all subsequent angles
    for i in np.arange(firstkey+1, len(out)):
        dphi = phi[i] - phi[i-1]
        if dphi < 0:
            dphi += 2.*np.pi
        out[i] = out[i-1] + dphi/2.0

    # set all previous angles to be the bar angle
    for i in np.arange(0, firstkey):
        out[i] = first_bar_angle

    return out

def get_sorted_keys(dat):
    keys = list(dat.keys())
    # only keep keys that are snapshot keys
    keys = [k for k in keys if 'snapshot' in k]

    # extract and sort indices
    indices = [int(re.findall(r'\d?\d?\d\d\d', k)[0]) for k in keys]
    sorted_arg = np.argsort(indices)
    keys_sorted = [keys[i] for i in sorted_arg]

    Rlist = np.array([np.array(fourier[k]['Rlist']) for k in tqdm(keys_sorted)])
    A2r = np.array([np.array(fourier[k]['A2r']) for k in tqdm(keys_sorted)])
    A2i = np.array([np.array(fourier[k]['A2i']) for k in tqdm(keys_sorted)])
    A0 = np.array([np.array(fourier[k]['A0']) for k in tqdm(keys_sorted)])
    return keys_sorted

def get_A2_angle(dat, keys):
    Rlist = np.array([np.array(dat[k]['Rlist']) for k in keys])
    A2r = np.array([np.array(dat[k]['A2r']) for k in keys])
    phi = np.arctan2(A2i, A2r)
    phi = phi[:,Rbin]
    R_at_Rbin = Rlist[:,Rbin]

    return time_list, R_at_Rbin, phi

def evaluate_polynomial(pfit, n, time, bar_angle_firstkey, firstkey):
    pfit_n = pfit[n]
    poly_bar_angle = np.zeros(len(time))
    poly_pattern_speed = np.zeros(len(time))

    for i in range(n+1):
        ba = pfit_n[i] * time ** (n-i)
        poly_bar_angle[firstkey:] += ba[firstkey:]
        ps = (n-i) * pfit_n[i] * time**(n-1-i)
        poly_pattern_speed[firstkey:] += ps[firstkey:]

    poly_bar_angle[:firstkey] += bar_angle_firstkey

    poly_pattern_speed = poly_pattern_speed / u.Myr
    poly_pattern_speed = poly_pattern_speed.to_value(u.km/u.s/u.kpc)

    return poly_bar_angle, poly_pattern_speed

def master_bar_angle(fname, name, output_dir='data/'):
    # try loading snapshot
    dat = h5.File(fname, mode='r')
    out = {}

    keys = get_sorted_keys(dat)
    time, R, phi = get_A2_angle(dat, keys)
    bar_angle = get_bar_angle(phi, firstkey)

    pattern_speed = np.gradient(bar_angle, time) / u.Myr
    pattern_speed = pattern_speed.to_value(u.km/u.s/u.kpc)

    pfit = [np.polyfit(time[firstkey:], bar_angle[firstkey:], i) for i in range(nmax)]
    
    out['time'] = time
    out['firstkey'] = firstkey
    out['R'] = R
    out['phi'] = phi
    out['bar_angle'] = bar_angle
    out['pattern_speed'] = pattern_speed
    out['pfit'] = pfit

    # now evaluate the polynomial for each fit and save the result
    out['poly_eval'] = {}
    for n in range(nmax):
        poly_bar_angle, poly_pattern_speed = evaluate_polynomial(pfit, n, time, bar_angle[firstkey], firstkey)

        out['poly_eval'][n] = (poly_bar_angle, poly_pattern_speed)

    pickle.dump(out, open(output_dir + 'bar_angle_'+name+'.p', 'wb'))


if __name__ == '__main__':

    basepath = '../fourier_component/data'

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_g2 = 'fid-disp1.0-fg0.2'
    fid_g3 = 'fid-disp1.0-fg0.3'
    fid_g4 = 'fid-disp1.0-fg0.4'
    fid_g5 = 'fid-disp1.0-fg0.5'    
    fid_d7_g3 = 'fid-disp0.7-fg0.3'
    fid_d5_g3 = 'fid-disp0.5-fg0.3'
    fid_g3_nB = 'fid-disp1.0-fg0.3-noBulge' 
    fid_g1_da = 'fid-disp1.0-fg0.1-diskAcc1.0'
    fid_g1_da_am = 'fid-disp1.0-fg0.1-diskAcc1.0-decAngMom' 
    fid_g1_corona = 'fid-disp1.0-fg0.1-corona'
    fid_g1_coronaRot = 'fid-disp1.0-fg0.1-coronaRot'
    fid_g1_coronaMat = 'fid-disp1.0-fg0.1-corona-Matthew'
    fid_g1_coronaMat4 = 'fid-disp1.0-fg0.1-corona-Matthew-MHG0.004'
    
    fid_g1_fixed1kpc = 'fid-disp1.0-fixedDisk-core1kpc'
    fid_g1_fixed2kpc = 'fid-disp1.0-fixedDisk-core2kpc'
    fid_g1_fixed3kpc = 'fid-disp1.0-fixedDisk-core3kpc'
    fid_g1_fixed4kpc = 'fid-disp1.0-fixedDisk-core4kpc' 
    fid_g1_fixed5kpc = 'fid-disp1.0-fixedDisk-core5kpc' 
    fid_g1_fixed6kpc = 'fid-disp1.0-fixedDisk-core6kpc' 
    fid_g1_dS_out_delay = 'fid-disp1.0-fg0.1-diskAGB-outer-delay1.0'


    pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), #(fid_g1, 'lvl3'),
                 #(fid_g2, 'lvl5'), (fid_g2, 'lvl4'), (fid_g2, 'lvl3'),
                 #(fid_g3, 'lvl5'), (fid_g3, 'lvl4'), (fid_g3, 'lvl3'),
                 #(fid_g4, 'lvl5'), (fid_g4, 'lvl4'),
                 #(fid_g5, 'lvl5'), (fid_g5, 'lvl4'),
                 (fid_g1_fixed1kpc, 'lvl5'), (fid_g1_fixed1kpc, 'lvl4'),
                 (fid_g1_fixed2kpc, 'lvl5'), (fid_g1_fixed2kpc, 'lvl4'),
                 (fid_g1_fixed3kpc, 'lvl5'), (fid_g1_fixed3kpc, 'lvl4'),
                 (fid_g1_fixed4kpc, 'lvl5'), (fid_g1_fixed4kpc, 'lvl4'),
                 (fid_g1_fixed5kpc, 'lvl5'), (fid_g1_fixed5kpc, 'lvl4'),
                 (fid_g1_fixed6kpc, 'lvl5'), (fid_g1_fixed6kpc, 'lvl4'),
                 (fid_g1_dS_out_delay, 'lvl5'), (fid_g1_dS_out_delay, 'lvl4'), (fid_g1_dS_out_delay, 'lvl3')]
                         
                 #(fid_g1_corona, 'lvl5'), (fid_g1_corona, 'lvl4'),
                 #(fid_g1_coronaRot, 'lvl5'), (fid_g1_coronaRot, 'lvl4'),
                 #(fid_g1_coronaMat, 'lvl5'), (fid_g1_coronaMat, 'lvl4'),
                 #(fid_g1_coronaMat4, 'lvl5'), (fid_g1_coronaMat4, 'lvl4')]
                 #(fid_d7_g3, 'lvl5'), (fid_d7_g3, 'lvl4'),
                 #(fid_d5_g3, 'lvl5'), (fid_d5_g3, 'lvl4'),
                 #(fid_g3_nB, 'lvl5'), (fid_g3_nB, 'lvl4'),
                 #(fid_g1_da, 'lvl5'), (fid_g1_da, 'lvl4'),
                 #(fid_g1_da_am, 'lvl5'), (fid_g1_da_am, 'lvl4')]
    
    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    # path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]

    fname_list = [basepath + '/fourier_' + name + '.hdf5' for name in name_list]

    for fname, name in zip(tqdm(fname_list), name_list):
        master_bar_angle(fname, name)
    
