#!/bin/sh
#SBATCH -p hernquist_ice,bigmem
##SBATCH -p itc_cluster,shared,conroy,hernquist
#SBATCH -J htrap 
#SBATCH -n 64
#SBATCH -N 1
#SBATCH -o logs/OUTPUT.%j.out
#SBATCH -e logs/ERROR.%j.err
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=480G
#SBATCH -t 0-06:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_halo_trapped.py ${SLURM_NTASKS} $1

