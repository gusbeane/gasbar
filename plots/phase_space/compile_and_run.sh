#!/bin/bash

# compile
mpicc -g -ggdb -lm -lhdf5 -o compute_phase_space compute_phase_space.c
#gcc -g -ggdb -lm -lhdf5 -o compute_phase_space compute_phase_space.c

# run
mpirun -np 32 ./compute_phase_space Nbody lvl4

# check output
#for i in {0..31}
#do
#    h5diff data/Nbody-lvl4/phase_space_Nbody-lvl4.${i}.hdf5 data/Nbody-lvl4-test/phase_space_Nbody-lvl4.${i}.hdf5
#done

