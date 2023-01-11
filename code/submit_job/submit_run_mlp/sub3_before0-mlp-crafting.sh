#!/bin/bash
#SBATCH --job-name=crafting
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
cd /user/work/qh22492/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv
#echo ciao
python /user/work/qh22492/e4d/run.py --config /user/work/qh22492/e4d/configs/mlp_config/sub3_before0-loo_Crafting.yaml