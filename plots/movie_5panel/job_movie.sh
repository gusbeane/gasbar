#!/bin/sh
#SBATCH -p conroy 
##SBATCH --constraint=intel
#SBATCH -J movie 
#SBATCH -n 4 
#SBATCH -N 1
#SBATCH -o OUTPUT_frames.%j.out
#SBATCH -e ERROR_frames.%j.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=16000
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 7-00:00           # Runtime in D-HH:MM

source ../load-modules.sh
module load parallel

ulimit -c unlimited

seq 0 20 | parallel -j ${SLURM_NTASKS} python3 make_movie_5panel.py {}

