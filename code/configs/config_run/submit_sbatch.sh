#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=20:00:00
#SBATCH --mem=150000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --output=test
cd /user/work/qh22492/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv
#echo ciao
python /user/work/qh22492/e4d/run.py --config /user/work/qh22492/e4d/configs/config_run/CE_CE-mix_CLIP-mix_trx_0_vit32.yaml