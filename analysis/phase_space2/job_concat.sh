#!/bin/sh
#SBATCH -p itc_cluster,shared
##SBATCH --constraint=intel
#SBATCH -J phsp 
#SBATCH -n 24
#SBATCH -N 1
#SBATCH -o logs/OUTPUT_frames.%j.out
#SBATCH -e logs/ERROR_frames.%j.err
##SBATCH --exclusive
##SBATCH --mail-user=angus.beane@cfa.harvard.edu
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
##SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=7000
##SBATCH --mem=64G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 0-12:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 concat_phase_space.py $1 $2 ${SLURM_NTASKS}

