#!/bin/sh
#SBATCH -p shared,hernquist,itc_cluster,conroy
#SBATCH -J torque 
#SBATCH -c 1
#SBATCH -o logs_SMUGGLE-lvl3/OUTPUT.%A.%a.out
#SBATCH -e logs_SMUGGLE-lvl3/ERROR.%A.%a.err
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=12G
##SBATCH --array=0-1
#SBATCH --array=0-1599
#SBATCH -t 0-00:45           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_torques.py phantom-vacuum-Sg20-Rc3.5 lvl3 ${SLURM_ARRAY_TASK_ID} 

