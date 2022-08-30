#!/bin/bash

for I in 1 2 4 5 6 7 8 16 64
do
    sbatch job_varyI.sh $I
done
