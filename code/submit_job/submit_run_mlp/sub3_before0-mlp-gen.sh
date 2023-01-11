#!/bin/bash
#SBATCH --job-name=MMD_gaussian_0,1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --output=MMD_gaussian_0,1

cd /user/work/qh22492/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv
#echo ciao
python /user/work/qh22492/e4d/run.py --config /user/work/qh22492/e4d/configs/config/CE_CE-mix_CLIP-mix_trx_2_0,5.yaml