#!/usr/bin/env bash

export PATH=~/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate pylops-distributed

#export OMP_NUM_THREADS=12
#export MKL_NUM_THREADS=12
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/
python MDC_timing.py 8
python MDC_timing.py 4
python MDC_timing.py 2
python MDC_timing.py 1

