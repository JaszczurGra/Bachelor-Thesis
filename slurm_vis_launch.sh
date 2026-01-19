#!/bin/bash

#SBATCH --job-name=diffusion_vis
#SBATCH --output=log/vis_%j.out
#SBATCH --error=log/vis_%j.err
#SBATCH --partition=tesla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion

echo "Starting visualization..."

python src_model/visualize_model.py \
    --run_url "https://wandb.ai/j-boro-poznan-university-of-technology/Bachelor-Thesis-src_model/sweeps/5zjjjyz2/runs/cqh7b021" \
    --save \
    -n 9

echo "Visualization finished."