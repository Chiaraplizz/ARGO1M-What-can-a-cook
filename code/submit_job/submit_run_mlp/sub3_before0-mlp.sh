#!/bin/bash
#SBATCH --job-name=CE
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30:00:00
#SBATCH --mem=10000
#SBATCH --cpus-per-task=2
#SBATCH --exclude=gpu12
#SBATCH --gres=gpu:1
#SBATCH --exclude=gpu12
#SBATCH --partition gpu
#SBATCH --output=CE

cd /user/work/qh22492/
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ffcv
#echo ciao
python /user/work/qh22492/e4d/run.py --config /user/work/qh22492/e4d/configs/config/CE.yaml