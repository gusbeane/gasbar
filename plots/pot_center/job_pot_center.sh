#!/bin/sh
##SBATCH -p bigmem
#SBATCH -p hernquist,conroy
##SBATCH -p itc_cluster,shared
##SBATCH --constraint=intel
#SBATCH -J pot 
#SBATCH -n 48
#SBATCH -N 1
#SBATCH -o OUTPUT_frames.%j.out
#SBATCH -e ERROR_frames.%j.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
##SBATCH --mem-per-cpu=3900
#SBATCH --mem=120G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 7-00:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_pot_center.py ${SLURM_NTASKS} $1

