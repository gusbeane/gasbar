#!/bin/bash

for i in 3 4 5
do
    python3 make_movie_frames.py -s /n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo/galakos/lvl${i}/output/snapshot -f 400 -p PartType0 -o PartType0_lvl${i}_w15.pickle -w 15 -x 200 -y 200 -z 200 -m 1.0
    python3 make_movie_frames.py -s /n/scratchlfs/hernquist_lab/abeane/mwib_runs/arepo/galakos/lvl${i}/output/snapshot -f 400 -p PartType0 -o PartType0_lvl${i}_w6.pickle -w 6 -x 200 -y 200 -z 200 -m 1.0
done




