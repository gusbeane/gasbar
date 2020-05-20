#!/bin/bash

#rsync -azvp --progress 'cannon:/n/home01/abeane/starbar/plots/movie_5panel-fg/movies/*.mp4' movies/
rsync -azvp --progress 'cannon:/n/home01/abeane/starbar-correcth/plots/movie_5panel/movies/*.mp4' movies/
rsync -azvp --progress 'cannon:/n/home01/abeane/starbar-correcth/plots/movie_5panel/profiles/*.prof' profiles/

