#!/bin/bash

#SBATCH --job-name=prompt-mix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --output=trans-prompt-mix

cd /user/work/qh22492/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv

python /user/work/qh22492/e4d/run.py --config /user/work/qh22492/e4d/configs/transformer_config/sub3_before0-prompt-mix.yaml