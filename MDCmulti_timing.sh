#!/usr/bin/env bash

export PATH=~/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate pylops-distributed

#export OMP_NUM_THREADS=12
#export MKL_NUM_THREADS=12
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/

python MDCmulti_timing.py 2 1
python MDCmulti_timing.py 2 5
python MDCmulti_timing.py 2 10
python MDCmulti_timing.py 2 25
#python MDCmulti_timing.py 2 40
