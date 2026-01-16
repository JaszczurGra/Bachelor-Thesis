#!/bin/bash

#SBATCH --job-name=diffusion_planning_singular
#SBATCH --output=log/diffusion_planning_singular_%A_%a.out
#SBATCH --error=log/train_%j.err
#SBATCH --partition=tesla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-10:00:00



eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion

echo "Starting training..."
python src_model/train.py

echo "Training finished."