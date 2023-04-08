#!/bin/bash

export NAME=$1
export LVL=$2

for i in {0..1601}
do
    if [ ! -f ./data/${NAME}-${LVL}/tmp${i}/tmp255.hdf5 ]
    then
        #sbatch job_phase_space.sh $NAME $LVL $i
        echo $i
    fi
done

