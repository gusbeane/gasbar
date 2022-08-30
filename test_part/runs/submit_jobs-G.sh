#!/bin/bash

for G in 0 2 4 6 8 10 12 14 16 18 20 
do
    sbatch job_varyG.sh $G
done
