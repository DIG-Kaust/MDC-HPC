#!/usr/bin/env bash

#export OMP_NUM_THREADS=12
#export MKL_NUM_THREADS=12
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/

#python Marchenko3D.py 8 650 1 0 620 1 0 580
python Marchenko3D.py 4 650 11 50 550 11 50 500
#python Marchenko3D.py 4 650 41 20 200 71 20 200
