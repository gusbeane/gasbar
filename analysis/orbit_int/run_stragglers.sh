#!/bin/bash

export NAME=$1
export LVL=$2

for i in {0..1023}
do
    if [ ! -f ./data/${NAME}-${LVL}/freqs_${NAME}-${LVL}.${i}.npy ]
    then
        #sbatch job_orbit_int-Nbody-lvl3-single.sh $i
        sbatch job_orbit_int-SMUGGLE-lvl3-single.sh $i
        echo $i
    fi
done

