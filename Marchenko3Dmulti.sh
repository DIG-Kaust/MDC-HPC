#!/usr/bin/env bash

export PATH=~/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate pylops-distributed

#export OMP_NUM_THREADS=12
#export MKL_NUM_THREADS=12
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/

python Marchenko3Dmulti.py 4 650 71 20 200 41 20 200 0 2900 20

