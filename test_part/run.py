from integrate import integrate_particles
import arepo
import numpy as np
import sys

M = 100.698
a = 26.2432

A = 0.02
b = 0.28
vc = 235
ps = 40

Tmax = 0.25
dt = 0.01

I_at_rCR6 = float(sys.argv[1])
torque_gas = float(sys.argv[2])

ics_dir = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/test_part/ics/'
lvl = 'lvl4'

ics = arepo.Snapshot(ics_dir + lvl + '/MW_ICs.dat')
pos = np.copy(ics.part1.pos)
vel = np.copy(ics.part1.vel)
mres = ics.MassTable[1]

out = integrate_particles(pos, vel, mres, M, a, A, b, vc, ps, Tmax=Tmax, dt=dt, torque_gas=0.0, verbose=True)

np.save('out.npy', out)
