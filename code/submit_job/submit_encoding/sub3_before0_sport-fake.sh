#!/bin/bash

#SBATCH --job-name=Sport
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --cpus-per-task=10

source ~/miniconda3/etc/profile.d/conda.sh

conda activate ffcv
python ../../scripts/dataset_ffcv_encode.py --config /user/work/qh22492/e4d//configs/mlp_config/sub3_before0-loo_Sport-fake.yaml --split train
