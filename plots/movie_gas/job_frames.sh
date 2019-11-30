#!/bin/sh
#SBATCH -p hernquist,itc_cluster
#SBATCH -J frames 
#SBATCH -n 16
#SBATCH -N 1
#SBATCH -o OUTPUT_frames.%j.out
#SBATCH -e ERROR_frames.%j.err
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

echo $OMP_NUM_THREADS
export OMP_NUM_THREADS=$SLURM_NTASKS

for i in 3-hernquist 4 5
do
    python3 make_movie_frames.py -s /n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo/galakos/lvl${i}/output/snapshot -f 400 -p PartType0 -o PartType0_lvl${i}_w15.pickle -w 15 -x 200 -y 200 -z 200 -m 1.0
    python3 make_movie_frames.py -s /n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo/galakos/lvl${i}/output/snapshot -f 400 -p PartType0 -o PartType0_lvl${i}_w6.pickle -w 6 -x 200 -y 200 -z 200 -m 1.0
done

