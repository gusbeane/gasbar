#!/bin/sh
#SBATCH -p hernquist 
#SBATCH -J phspce 
#SBATCH -n 6
#SBATCH -N 1
#SBATCH -o OUTPUT_frames.%j.out
#SBATCH -e ERROR_frames.%j.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=240G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 7-00:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

mpicc -g -ggdb -lm -lhdf5 -o compute_phase_space compute_phase_space.c

mpirun -np ${SLURM_NTASKS} ./compute_phase_space $1 $2

#mpirun -np ${SLURM_NTASKS}  ./compute_phase_space phantom-vacuum-Sg20-Rc3.5 lvl3-rstHalo
#mpirun -np ${SLURM_NTASKS}  ./compute_phase_space phantom-vacuum-Sg20-Rc3.5 lvl3
#mpirun -np ${SLURM_NTASKS}  ./compute_phase_space phantom-vacuum-Sg20-Rc3.5 lvl4

