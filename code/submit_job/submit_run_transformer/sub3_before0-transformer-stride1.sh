#!/bin/bash

#SBATCH --job-name=sub3_before2-layer4-stride1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --output=R-sub3_before2-layer4-stride1.out

cd /user/work/qh22492/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv

python /user/work/qh22492/e4d/run.py --config /user/work/qh22492/e4d/configs/transformer_config/sub3_before0-layer4-stride1.yaml