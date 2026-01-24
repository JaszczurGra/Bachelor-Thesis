#!/bin/bash

#SBATCH --job-name=diffusion_vis
#SBATCH --output=log/vis_%j.out
#SBATCH --error=log/vis_%j.err
#SBATCH --partition=tesla
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=0-23:59:59

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion

echo "Starting visualization..."

python src_model/visualize_model.py \
    --run_url "$1" \
    --save \
    --batch_size "$2"\
    --custom_dataset "$3"
  
#grouchy_penguin_23-01-2026_00:43:35
echo "Visualization finished."