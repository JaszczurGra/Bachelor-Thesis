#!/bin/bash


JOB_ID=$(sbatch --parsable run_parallel_slurm.sh "$@")
echo "Submitted array job: $JOB_ID"

POST_JOB_ID=$(sbatch --parsable --dependency=afterok:${JOB_ID} combine.sh "$@")
echo "Submitted post-processing job: $POST_JOB_ID (waits for $JOB_ID)"

echo "Monitor jobs with: squeue -u $USER"