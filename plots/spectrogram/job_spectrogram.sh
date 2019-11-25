#!/bin/sh
#SBATCH -p itc_cluster
#SBATCH -J spectro 
#SBATCH -n 1 
#SBATCH -N 1
#SBATCH -o OUTPUT_spectrogram.%j.out
#SBATCH -e ERROR_spectrogram.%j.err
##SBATCH --exclusive
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=4000
##SBATCH -t 4-00:00           # Runtime in D-HH:MM
#SBATCH -t 7-00:00           # Runtime in D-HH:MM

source ../../load-modules.sh

ulimit -c unlimited

python3 compute_spectrogram.py

