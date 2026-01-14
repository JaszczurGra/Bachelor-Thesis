#!/bin/bash

# --- SLURM JOB DEFINITION ---
#SBATCH --job-name=single_train_run
#SBATCH --output=log/train_%j.out  # %j will be replaced by the job ID
#SBATCH --error=log/train_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-10:00:00 # 10 hours

# --- ENVIRONMENT SETUP ---
# Activate your Conda environment
# Make sure the path to conda is correct for your system
eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion

# --- EXECUTE THE TRAINING SCRIPT ---
# This command directly runs your Python script.
# wandb will be activated by the wandb.init() call inside train.py
echo "Starting training..."
python src_model/train.py

echo "Training finished."