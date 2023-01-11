#!/bin/bash

#SBATCH --job-name=sub3_before0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=10

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv

python /user/work/qh22492/e4d/scripts/dataset_ffcv_encode.py --config /user/work/qh22492/e4d/configs/config/sub3_before0-new_slowfast-prova.yaml --split train
