#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --job-name=pytorch
#SBATCH --output=script.out
#SBATCH --error=script_error.out

module load DL_conda_3.7/3.7

python3 run.py
