#!/bin/bash

N=8
R=3
T=7200


if [ -z "$1" ]; then
    echo "Error: Map file required"
    exit 1
fi

ADJECTIVES=("fuzzy" "wiggly" "bouncy" "sleepy" "sneaky" "grumpy" "clumsy" "sassy" "sparkly" "grouchy" "zippy" "lazy")
ANIMALS=("grizzly" "penguin" "kangaroo" "sloth" "ferret" "badger" "llama" "otter" "dolphin" "eagle" "wolf" "panda")

RANDOM_ANIMAL="${ADJECTIVES[$RANDOM % ${#ADJECTIVES[@]}]}_${ANIMALS[$RANDOM % ${#ANIMALS[@]}]}_$(date +"%d-%m-%Y_%H:%M:%S")" 

echo "Running with animal name: $RANDOM_ANIMAL" and map folder: "$1"


#--time=${TIME_LIMIT}
JOB_ID=$(sbatch --parsable slurm_generate_folders.sh "$RANDOM_ANIMAL" "$1" "$N" "$R")
echo "Submitted folder generation job: $JOB_ID"

POST_JOB_ID=$(sbatch --parsable --dependency=afterok:${JOB_ID} run_parallel_slurm.sh "$RANDOM_ANIMAL" "$1" "$N" "$R")
echo "Submitted worker jobs: $POST_JOB_ID (waits for $JOB_ID)"

echo "Monitor jobs with: squeue -u $USER"

