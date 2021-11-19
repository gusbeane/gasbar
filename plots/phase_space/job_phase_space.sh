#!/bin/sh
#SBATCH -p itc_cluster,shared,hernquist,conroy
##SBATCH --constraint=intel
#SBATCH -J orbits 
#SBATCH -n 12
#SBATCH -N 1
#SBATCH -o OUTPUT_frames.%j.out
#SBATCH -e ERROR_frames.%j.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
##SBATCH --mem-per-cpu=5500
#SBATCH --mem=164G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 7-00:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_phase_space.py ${SLURM_NTASKS} $1

