#!/bin/bash

#SBATCH --job-name=start
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition gpu

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv

python ../scripts/dataset_ffcv_encode.py --config ../configs/mid.yaml --split train
python ../scripts/dataset_ffcv_encode.py --config ../configs/mid.yaml --split val
python ../scripts/dataset_ffcv_encode.py --config ../configs/mid.yaml --split test_sport
python ../scripts/dataset_ffcv_encode.py --config ../configs/mid.yaml --split test_knit
python ../scripts/dataset_ffcv_encode.py --config ../configs/mid.yaml --split test_cooking_tokyo