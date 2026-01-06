#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: Map file required"
    echo "Usage: bash run_combined_slurm.sh <output_dir> <map_file>"
    exit 1
fi

ADJECTIVES=("fuzzy" "wiggly" "bouncy" "sleepy" "sneaky" "grumpy" "clumsy" "sassy" "sparkly" "grouchy" "zippy" "lazy")
ANIMALS=("grizzly" "penguin" "kangaroo" "sloth" "ferret" "badger" "llama" "otter" "dolphin" "eagle" "wolf" "panda")

RANDOM_ANIMAL="${ADJECTIVES[$RANDOM % ${#ADJECTIVES[@]}]}_${ANIMALS[$RANDOM % ${#ANIMALS[@]}]}"

echo "Running with animal name: $RANDOM_ANIMAL" and map folder: "$1"


export SLURM_ARRAY_TASK_ID=0
./run_parallel_slurm.sh "$RANDOM_ANIMAL" "$1"
export SLURM_ARRAY_TASK_ID=1
./run_parallel_slurm.sh "$RANDOM_ANIMAL" "$1"



./combine.sh "$1"
