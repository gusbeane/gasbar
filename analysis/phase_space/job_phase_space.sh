#!/bin/sh
#SBATCH -p hernquist_ice,hernquist,itc_cluster,conroy,shared
##SBATCH --constraint=intel
#SBATCH -J phsp 
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -o logs/OUTPUT_phsp_%A.%a.out
#SBATCH -e logs/ERROR_phsp_%A.%a.err
##SBATCH --mail-user=angus.beane@cfa.harvard.edu
##SBATCH --mail-type=BEGIN
##SBATCH --mail-type=END
##SBATCH --mail-type=FAIL
#SBATCH --mem=8G
#SBATCH -t 60:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_phase_space.py $1 $2 $3 

