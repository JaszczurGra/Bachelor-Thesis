#!/bin/bash
# filepath: /home/jaszczurgra/Documents/Programing/Bachelor-Thesis/slurm_wandb_sweep.sh

#SBATCH --job-name=wandb_sweep
#SBATCH --output=log/sweep_%A_%a.out
#SBATCH --error=log/sweep_%A_%a.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=0-06:00:00
#SBATCH --array=0-9  # 10 agents running in parallel

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion

# Each SLURM array job runs as a W&B sweep agent
wandb agent <your_entity>/<your_project>/<sweep_id>