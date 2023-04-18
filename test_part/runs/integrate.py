import numpy as np
from numba import njit, prange
from numba.core import types
from numba.experimental import jitclass
from force import Hernquist_force, bar_force, MiyamotoNagaiDisk

spec = [
    ('time', types.float64[:]),
    ('total_torque', types.float64[:,:]),
    ('torque_gas', types.float64[:]),
    ('ps', types.float64[:]),
    ('M', types.float64),
    ('a', types.float64),
    ('A', types.float64),
    ('b', types.float64),
    ('vc', types.float64),
    ('mres', types.float64),
]



@jitclass(spec)
class output(object):
    def __init__(self):
        pass
#         key_list = []
#         for key, _ in spec:
#             key_list.append(key)

#             self.key_list = key_list

@njit(parallel=True)
def integrate_particles(pos, vel, mres, M, a, A, b, vc, ps, disk_M, disk_a, disk_b, dt=0.1, Tmax=1.0,
                        torque_gas=20.0, I_at_rCR6=2.0, G=43007.1, verbose=False):
    T = 0.0
    
    ang = 0.0
    
    acc_halo = Hernquist_force(pos, M, a, G)
    acc_bar = bar_force(pos, A, b, vc, ps, ang)
    acc_disk = MiyamotoNagaiDisk(pos, disk_M, disk_a, disk_b, G)
    acc = acc_halo + acc_bar + acc_disk
    
    Nint = int(Tmax/dt)+1
    print('Nint=', Nint)
    init_ps = ps
    
    # Output arrays
    total_torque = np.zeros((Nint, 3))
    time_out = np.zeros(Nint)
    ps_out = np.zeros(Nint)
    torque_gas_list = np.zeros(Nint)
    
    torque_gas0 = torque_gas
    ps0 = ps

    N = pos.shape[0]
    
    for i in range(Nint):
        # Analysis
        torque = mres * np.cross(pos, acc)
        total_torque[i] = np.sum(torque, axis=0)
        ps_out[i] = ps
        time_out[i] = T
        torque_gas_list[i] = torque_gas

        # Print
        if verbose:
            print('i=', i, ' T=', T, 'total_torque=', total_torque[i], 'ps=', ps)
        
        # First half-kick and drift
        for j in prange(N):
            for k in range(3):
                vel[j][k] += acc[j][k] * dt / 2.
                pos[j][k] += vel[j][k] * dt
        
        rCR = vc/ps
        I = I_at_rCR6 * (rCR/6.0)**2
        
        # Bar half-kick
        ps += - total_torque[i][2]/I * dt / 2.0 # constant ps
        ps += torque_gas/I * dt / 2.0
        
        #if torque_gas0 > 0:
        #    torque_gas = torque_gas0 + (40-ps)
        #else:
        #    torque_gas = torque_gas0

        # Bar drift
        ang += ps * dt
        
        # Force computation
        acc_halo = Hernquist_force(pos, M, a, G)
        acc_bar = bar_force(pos, A, b, vc, ps, ang)
        acc_disk = MiyamotoNagaiDisk(pos, disk_M, disk_a, disk_b, G)
        acc = acc_halo + acc_bar + acc_disk
        
        # Second half-kick
        for j in prange(N):
            for k in range(3):
                vel[j][k] += acc[j][k] * dt / 2.
        
        rCR = vc/ps
        I = I_at_rCR6 * (rCR/6.0)**2
        
        ps += - total_torque[i][2]/I * dt / 2.0 # constant ps
        ps += torque_gas/I * dt / 2.0
        
        T += dt

    out = output()
    out.time = time_out
    out.total_torque = total_torque
    out.torque_gas = torque_gas_list
    out.ps = ps_out

    out.M = M
    out.a = a
    out.A = A
    out.b = b
    out.vc = vc
    out.mres = mres

    return out
