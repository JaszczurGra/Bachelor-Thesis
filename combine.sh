#!/bin/bash

#SBATCH --job-name=MotionPlanningDatasetgenCombining
#SBATCH --output=log/%j.out
#SBATCH --error=log/%j.err
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3000
#SBATCH --time=0-0:30:59
#SBATCH --array=0


python src/combine_runs.py --save "$1" 
