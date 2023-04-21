#!/bin/sh
##SBATCH -p itc_cluster,shared
#SBATCH -p shared,hernquist,hernquist_ice,itc_cluster,conroy 
##SBATCH --constraint=intel
#SBATCH -J Ssphsp 
#SBATCH -c 1
#SBATCH --array=0-1600
##SBATCH --array=0-15
#SBATCH -o logs_SMUGGLEstar-lvl4/OUTPUT.%A.%a.out
#SBATCH -e logs_SMUGGLEstar-lvl4/ERROR.%A.%a.err
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
##SBATCH --mem-per-cpu=5500
#SBATCH --mem=8G
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 15:00          # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_phase_space.py phantom-vacuum-Sg20-Rc3.5-star lvl4 ${SLURM_ARRAY_TASK_ID}

