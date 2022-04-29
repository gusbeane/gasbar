import numpy as np
import arepo
import sys
from tqdm import tqdm
from scipy.optimize import minimize
from numba import jit

from joblib import Parallel, delayed

m_p = 1.67262178e-24 # g
k = 1.38065e-16 # cgs

G = 43007.1

nproc=6

def compute_u_from_T(T):
    # T in K
    if T > 1e4:
        mu = 10.0/17.0 #Assume fully ionised, ignore metals
    else:
        mu = 50.0/41.0 #Assume fully neutral, ignore metals

    u = (3.0 * k * T) / (2.0 * mu * m_p) # cgs units
    u /= 1e10 # code units
    return u

def _Q1_of_k_(k, kappa, sigma, dens):
    return (kappa**2 + (k*sigma)**2) / (2. * np.pi * G * k * dens)

@jit(nopython=True)
def _Q2_of_k_(k, cs, kappa, sigmaR, sigmaStar, sigmaGas):
    Qg = (kappa**2 + (k*cs)**2) / (2. * np.pi * G * k * sigmaGas)
    Qs = (kappa**2 + (k*sigmaR)**2) / (2. * np.pi * G * k * sigmaStar)
    return 1./(1./Qg + 1./Qs)

def _toomre2_(kmin, kmax, cs, kappa, sigmaR, sigmaStar, sigmaGas, returnk=False):
    fun_list = []
    x_list = []
    # print('cs=', cs, 'kappa=', kappa, 'sigmaR=', sigmaR, 'sigmaStar=', sigmaStar, 'sigmaGas=', sigmaGas)
    for kguess in np.logspace(np.log10(kmin), np.log10(kmax), 10):
        ans = minimize(_Q2_of_k_, kguess, (cs, kappa, sigmaR, sigmaStar, sigmaGas), bounds=((kmin, kmax),))
        fun_list.append(ans.fun[0])
        x_list.append(ans.x)

    fun = np.min(fun_list)
    x = x_list[np.argmin(fun)]

    if returnk:
        return fun, x
    else:
        return fun

def _toomre_star_(kappa, sigmaR, sigmaStar):
    return kappa * sigmaR / (3.36 * G * sigmaStar)

def _match_toomre_minimize_(sigmaR, Q, kmin, kmax, cs, kappa, sigmaStar, sigmaGas):
    Q2 = _toomre2_(kmin, kmax, cs, kappa, sigmaR, sigmaStar, sigmaGas)
    return np.abs(Q - Q2)

def _match_toomre_(kmin, kmax, Q, cs, kappa, sigmaStar, sigmaGas, sigmaR):
    fun_list = []
    x_list = []
    for sigmaR_guess in np.logspace(-4, 2.5, 8):
        ans = minimize(_match_toomre_minimize_, sigmaR_guess, (Q, kmin, kmax, cs, kappa, sigmaStar, sigmaGas), bounds=((0, 1000),))
        x_list.append(ans.x[0])
        fun_list.append(ans.fun)
    
    return x_list[np.argmin(fun_list)]


def compute_two_comp_toomre(path, name, output_dir='data/'):
    # try loading snapshot
    dat = np.genfromtxt(path+'/ICs/MW_ICs.vc2comp', names=True)

    gamma = 5./3.
    u = compute_u_from_T(1e4)
    cs2 = gamma * (gamma-1) * u
    cs = np.sqrt(cs2)

    kmin = 0.06
    kmax = 6000.0


    Q_star = []
    Q2_list = []
    Qm1k_star = []
    Qm1k_gas = []

    for kap, sR, sS, sG in zip(tqdm(dat['kappa']), dat['sigmaR'], dat['sigmaStar'], dat['sigmaGas']):
        this_Qstar = _toomre_star_(kap, sR, sS)
        Q_star.append(this_Qstar)

        this_Q2, k = _toomre2_(kmin, kmax, cs, kap, sR, sS, sG, returnk=True)
        Q2_list.append(this_Q2)

        Qm1k_star.append(float(_Q1_of_k_(k, kap, sR, sS)))
        Qm1k_gas.append(float(_Q1_of_k_(k, kap, cs, sG)))
    

    # Q_star = Parallel(n_jobs=nproc) (delayed(_toomre_star_)(k, sR, sS) for k, sR, sS in zip(tqdm(dat['kappa']), dat['sigmaR'], dat['sigmaStar']))
    # Q2_list = Parallel(n_jobs=nproc) (delayed(_toomre2_)(kmin, kmax, np.sqrt(cs2), k, sR, sS, sG) for k, sR, sS, sG in zip(tqdm(dat['kappa']), dat['sigmaR'], dat['sigmaStar'], dat['sigmaGas']))
    
    sigmaR_match = Parallel(n_jobs=nproc) (delayed(_match_toomre_)(kmin, kmax, Q, np.sqrt(cs2), k, sS, sG, sR) for Q, k, sS, sG, sR in zip(tqdm(Q_star), dat['kappa'], dat['sigmaStar'], dat['sigmaGas'], dat['sigmaR']))

    Q_at_sigmaR_match = Parallel(n_jobs=nproc) (delayed(_toomre2_)(kmin, kmax, np.sqrt(cs2), k, sR, sS, sG) for k, sR, sS, sG in zip(tqdm(dat['kappa']), sigmaR_match, dat['sigmaStar'], dat['sigmaGas']))

    fR_list = np.square(sigmaR_match/dat['sigmaR'])

    np.save(output_dir+name+'-Q2.npy', np.transpose([dat['R'], Q_star, Q2_list, fR_list, Q_at_sigmaR_match, Qm1k_star, Qm1k_gas]))


if __name__ == '__main__':

    basepath = '../../runs/'

    fid_g1 = 'fid-disp1.0-fg0.1'
    fid_d15_g1 = 'fid-disp1.5-fg0.1'

    # look to see if we are on my macbook or on the cluster
    if sys.platform == 'darwin':
        pair_list = [(fid_g1, 'lvl5'), (fid_d15_g1, 'lvl5')]
    else:
        pair_list = [(fid_g1, 'lvl5'), (fid_g1, 'lvl4'), (fid_g1, 'lvl3')]

    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + p[0] + '/' + p[1] for p in pair_list]
         
    #compute_two_comp_toomre('fid-fg0.1-Rcore1.0/', 'fid-fg0.1-Rcore1.0')
    #compute_two_comp_toomre('fid-fg0.1-Rcore1.0-MB0.004/', 'fid-fg0.1-Rcore1.0-MB0.004')
    compute_two_comp_toomre('fid-fg0.2-Rcore1.0-MB0.004/', 'fid-fg0.2-Rcore1.0-MB0.004')
    #compute_two_comp_toomre('fid-fg0.1-Rcore1.0-noMB/', 'fid-fg0.1-Rcore1.0-noMB')
    #for path, name in zip(path_list, name_list):
        #compute_two_comp_toomre(path, name)
    
