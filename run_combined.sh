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


export SLURM_ARRAY_TASK_ID=0
./run_parallel_slurm.sh "$1" "$2"
export SLURM_ARRAY_TASK_ID=1
./run_parallel_slurm.sh "$1" "$2"




./combine.sh "$1"


