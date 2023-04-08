from integrate import integrate_particles, spec
import arepo
import numpy as np
import sys
import pickle

# Units are given in kpc, km/s, and 1E10 Msun

# Halo properties
halo_M = 100.698
halo_a = 26.2432

# Disk properties
disk_a = 2.67074
disk_b = disk_a * 0.12
disk_M = 4.8

# Bar properties
A = 0.02
b = 0.28
vc = 235
ps = 40

Tmax = 5.0
dt = 0.01

I_at_rCR6 = float(sys.argv[1])
torque_gas = float(sys.argv[2])
fout = sys.argv[3]

ics_dir = '/n/holylfs05/LABS/hernquist_lab/Users/abeane/gasbar/test_part/ics/'
lvl = 'lvl4'

ics = arepo.Snapshot(ics_dir + lvl + '/MW_ICs.dat')
pos = np.copy(ics.part1.pos)
vel = np.copy(ics.part1.vel)
mres = ics.MassTable[1]

out = integrate_particles(pos, vel, mres, halo_M, halo_a, 
                          A, b, vc, ps,
                          disk_M, disk_a, disk_b,
                          Tmax=Tmax, dt=dt, torque_gas=torque_gas,
                          I_at_rCR6=I_at_rCR6,
                          verbose=True)

out_dict = {}
for key,_ in spec:
    out_dict[key] = getattr(out, key)

np.save(fout, out_dict)
