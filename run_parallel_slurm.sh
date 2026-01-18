#!/bin/bash

#SBATCH --job-name=MP_worker_dataset
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1000
#SBATCH --time=0-6:59:59
#SBATCH --array=0-159

eval "$(/mnt/storage_6/project_data/pl0467-01/soft/miniconda3/bin/conda shell.bash hook)"
conda activate planning_diffusion

TASK_ID=${SLURM_ARRAY_TASK_ID}

python src/parallel_generation.py --save "$1" -n "$3" -t 7200 -r "$4" --run_id ${TASK_ID} --map "$2" 


