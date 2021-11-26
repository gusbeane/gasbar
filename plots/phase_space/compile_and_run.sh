#!/bin/bash

# compile
gcc -g -ggdb -lhdf5 -o compute_phase_space compute_phase_space.c

# run
./compute_phase_space Nbody lvl4
