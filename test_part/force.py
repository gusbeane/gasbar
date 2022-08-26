import numpy as np
from numba import njit, prange

@njit(parallel=True)
def Hernquist_force(pos, M, a, G):
    N = pos.shape[0]
    mGM = - G * M
    acc = np.zeros(pos.shape)
    for i in prange(N):
        r = np.sqrt(pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1] + pos[i][2]*pos[i][2])
        acc[i][0] = mGM * pos[i][0] / (r * (a + r)**2)
        acc[i][1] = mGM * pos[i][1] / (r * (a + r)**2)
        acc[i][2] = mGM * pos[i][2] / (r * (a + r)**2)
        
    return acc

@njit(parallel=True)
def Hernquist_pot(pos, M, a, G):
    N = pos.shape[0]
    mGM = - G * M
    pot = np.zeros(N)
    for i in prange(N):
        r = np.sqrt(pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1] + pos[i][2]*pos[i][2])
        pot[i] = mGM / (r + a)
        
    return pot

@njit
def _compute_phi_r(A, vc, r, rCR, b):
    ans = - A * vc**2 / 2
    ans *= (r/rCR)**2
    ans *= ((b+1.)/(b+r/rCR))**5.
    return ans

@njit
def _compute_partial_phi_r(A, vc, r, rCR, b):
    term1 = - A * vc**2 * (r/rCR)**2 * ((b+1)/(b+r/rCR))**5
    term2 = (-A*vc**2/2) * (5 * ((b+1)/(b+r/rCR))**4) * (-((b+1)*rCR)/(r+rCR)**2)
    
    return term1 + term2

@njit(parallel=True)
def bar_pot(pos, A, b, vc, ps, ang):
    N = pos.shape[0]
    pot = np.zeros(N)
    rCR = vc / ps
    for i in range(N):
        r = np.sqrt(pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1] + pos[i][2]*pos[i][2])
        theta = np.arccos(pos[i][2]/r)
        phi = np.arctan2(pos[i][1], pos[i][0])
        
        phi_r = _compute_phi_r(A, vc, r, rCR, b)
        
        pot[i] = phi_r * np.sin(theta)**2 * np.cos(2.*(phi - ang))
    
    return pot

@njit(parallel=False)
def bar_force(pos, A, b, vc, ps, ang):
    acc = np.zeros(pos.shape)
    rCR = vc / ps
    N = pos.shape[0]
    
    for i in range(N):
        
        r = np.sqrt(pos[i][0]*pos[i][0] + pos[i][1]*pos[i][1] + pos[i][2]*pos[i][2])
        
        theta = np.arccos(pos[i][2]/r)
        phi = np.arctan2(pos[i][1], pos[i][0])
        
        phi_r = _compute_phi_r(A, vc, r, rCR, b)
        
        acc_r = -(1./r) * ((2*b-3*r/rCR)/(b+r/rCR)) * phi_r
        acc_r = - _compute_partial_phi_r(A, vc, r, rCR, b)
        acc_r *= np.sin(theta)**2 * np.cos(2.*(phi - ang))
        
        acc_theta = - phi_r * 2.*np.sin(theta)*np.cos(theta) * np.cos(2.*(phi-ang)) / r
        
        acc_phi = - phi_r * np.sin(theta) * (-2 * np.sin(2.*(phi-ang))) / r
        
        acc[i][0] = np.sin(theta)*np.cos(phi)*acc_r + np.cos(theta)*np.cos(phi)*acc_theta - np.sin(phi)*acc_phi
        acc[i][1] = np.sin(theta)*np.sin(phi)*acc_r + np.cos(theta)*np.sin(phi)*acc_theta + np.cos(phi)*acc_phi
        acc[i][2] = np.cos(theta) * acc_r - np.sin(theta) * acc_theta
        
    return acc
