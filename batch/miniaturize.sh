#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --mem=7GB
#SBATCH --time=02:00:00

source ~/.profile_minigraphs

GRAPH_NAME=$1
SIZE=$2
DENSITY=$3

mpiexec -n 6 python -u ../src/miniaturize.py "$GRAPH_NAME" "$SIZE" "$DENSITY"
