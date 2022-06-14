#!/bin/sh
#SBATCH -p itc_cluster,shared,conroy
#SBATCH -J oiS50 
#SBATCH -c 1
##SBATCH --array=0-1023
##SBATCH --array=0-15
#SBATCH -o logs_SMUGGLE-lvl3/OUTPUT.%A.%a.out
#SBATCH -e logs_SMUGGLE-lvl3/ERROR.%A.%a.err
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem=12G
#SBATCH -t 0-04:00           # Runtime in D-HH:MM

source ../load-modules.sh

ulimit -c unlimited

python3 compute_orbit_integrations.py phantom-vacuum-Sg20-Rc3.5 lvl3 300 ${1}

