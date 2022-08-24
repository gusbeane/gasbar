#!/bin/sh
##SBATCH -p itc_cluster,shared
#SBATCH -p serial_requeue
##SBATCH --constraint=intel
#SBATCH -J phsp 
#SBATCH -c 1
#SBATCH --array=0-1600
##SBATCH --array=0-15
#SBATCH -o logs_Nbody-lvl3/OUTPUT.%A.%a.out
#SBATCH -e logs_Nbody-lvl3/ERROR.%A.%a.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
##SBATCH --mem-per-cpu=5500
#SBATCH --mem=8G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 00:05           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_phase_space.py Nbody lvl3 ${SLURM_ARRAY_TASK_ID}

