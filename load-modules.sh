#!/bin/bash

module purge
module load gcc/7.1.0-fasrc01
module load openmpi/2.1.0-fasrc02
module load gsl
module load hdf5
module load python/3.6.3-fasrc02
module load ffmpeg/4.0.2-fasrc01

export PYTHONPATH={$PYTHONPATH}:/n/home01/abeane/python-code

