#!/bin/sh
#SBATCH -p itc_cluster,shared
##SBATCH --constraint=intel
#SBATCH -J phsp 
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o logs/OUTPUT_frames.%j.out
#SBATCH -e logs/ERROR_frames.%j.err
##SBATCH --exclusive
##SBATCH --mail-user=angus.beane@cfa.harvard.edu
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
##SBATCH --mail-type=FAIL
##SBATCH --mem-per-cpu=5500
#SBATCH --mem=8G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 01:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_phase_space.py $1 $2 $3

