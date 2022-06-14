#!/bin/bash

rsync -azvp --progress 'cannon:/n/home01/abeane/starbar/plots/fourier_component/data/*.hdf5' data/
