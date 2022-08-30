#!/bin/sh
#SBATCH -p shared,itc_cluster,conroy 
#SBATCH -J sam_vI 
#SBATCH -n 1
#SBATCH -N 1 
#SBATCH --ntasks-per-node=1
#SBATCH -o logs/OUTPUT.%j.out
#SBATCH -e logs/ERROR.%j.err
#SBATCH --mail-user=angus.beane@cfa.harvard.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mem-per-cpu=3900
#SBATCH -t 00-00:30           # Runtime in D-HH:MM

source ./load-modules.sh

export I=$1

python3 run.py $I 0.0 output-Ivar/out_I${I}.npy

