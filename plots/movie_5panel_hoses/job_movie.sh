#!/bin/sh
#SBATCH -p shared,conroy,hernquist,itc_cluster
#SBATCH -J movie 
#SBATCH -n 17
#SBATCH -N 1
#SBATCH -o OUTPUT_frames.%j.out
#SBATCH -e ERROR_frames.%j.err
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=8000
#SBATCH -t 1-00:00           # Runtime in D-HH:MM

source ../load-modules.sh
module load parallel

ulimit -c unlimited

seq 0 30 | parallel -j ${SLURM_NTASKS} python3 make_movie_5panel.py {}

