#!/usr/bin/env bash

export PATH=~/miniconda3/bin:$PATH
eval "$(conda shell.bash hook)"
conda activate pylops-distributed

#export OMP_NUM_THREADS=12
#export MKL_NUM_THREADS=12
export STORE_PATH=/project/fsenter/mrava/Marchenko3D/

python MDD3D.py 4 0 2900 20 Ss
