#!/bin/sh
#SBATCH -p itc_cluster,shared,conroy
##SBATCH --constraint=intel
#SBATCH -J torque 
#SBATCH -n 12
#SBATCH -N 1
#SBATCH -o logs/OUTPUT_%x.%j.out
#SBATCH -e logs/ERROR_%x.%j.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=180G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 0-04:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_torques.py ${SLURM_NTASKS} $1

