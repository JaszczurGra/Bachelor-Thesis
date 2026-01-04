#!/bin/bash


if [ -z "$1" ]; then
    echo "Error: Output directory required"
    echo "Usage: bash run_combined_slurm.sh <output_dir> <map_file>"
    exit 1
fi
if [ -z "$2" ]; then
    echo "Error: Map file required"
    echo "Usage: bash run_combined_slurm.sh <output_dir> <map_file>"
    exit 1
fi


JOB_ID=$(sbatch --parsable run_parallel_slurm.sh "$1" "$2")
echo "Submitted array job: $JOB_ID"

POST_JOB_ID=$(sbatch --parsable --dependency=afterok:${JOB_ID} combine.sh "$1")
echo "Submitted post-processing job: $POST_JOB_ID (waits for $JOB_ID)"

echo "Monitor jobs with: squeue -u $USER"

