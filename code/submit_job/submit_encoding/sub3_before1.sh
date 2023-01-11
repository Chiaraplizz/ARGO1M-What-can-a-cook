#!/bin/bash

#SBATCH --job-name=sub3_before1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=100000
#SBATCH --cpus-per-task=10

conda activate ffcv
python ../scripts/dataset_ffcv_encode.py --config ../configs/sub3_before1.yaml --split train
python ../scripts/dataset_ffcv_encode.py --config ../configs/sub3_before1.yaml --split val
python ../scripts/dataset_ffcv_encode.py --config ../configs/sub3_before1.yaml --split test_sport
python ../scripts/dataset_ffcv_encode.py --config ../configs/sub3_before1.yaml --split test_knit
python ../scripts/dataset_ffcv_encode.py --config ../configs/sub3_before1.yaml --split test_cooking_tokyo