#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Map file required"
    echo "Usage: bash run_combined_slurm.sh <output_dir> <map_file>"
    exit 1
fi

ADJECTIVES=("fuzzy" "wiggly" "bouncy" "sleepy" "sneaky" "grumpy" "clumsy" "sassy" "sparkly" "grouchy" "zippy" "lazy")
ANIMALS=("grizzly" "penguin" "kangaroo" "sloth" "ferret" "badger" "llama" "otter" "dolphin" "eagle" "wolf" "panda")

RANDOM_ANIMAL="${ADJECTIVES[$RANDOM % ${#ADJECTIVES[@]}]}_${ANIMALS[$RANDOM % ${#ANIMALS[@]}]}"

echo "Running with animal name: $RANDOM_ANIMAL"


JOB_ID=$(sbatch --parsable run_parallel_slurm.sh "$RANDOM_ANIMAL" "$1")
echo "Submitted array job: $JOB_ID"

POST_JOB_ID=$(sbatch --parsable --dependency=afterok:${JOB_ID} combine.sh "$RANDOM_ANIMAL")
echo "Submitted post-processing job: $POST_JOB_ID (waits for $JOB_ID)"

echo "Monitor jobs with: squeue -u $USER"

