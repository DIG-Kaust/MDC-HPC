#!/usr/bin/env bash

#export OMP_NUM_THREADS=12
#export MKL_NUM_THREADS=12
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/

python MDCmulti_timing.py 2 1
python MDCmulti_timing.py 2 5
python MDCmulti_timing.py 2 10
python MDCmulti_timing.py 2 25
#python MDCmulti_timing.py 2 40
