#!/bin/bash
#SBATCH --ntasks=6
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=02:00:00

source ~/.profile_minigraphs

GRAPH_NAME=$1
SIZE=$2

mpiexec -n 6 python -u ../src/miniaturize.py "$GRAPH_NAME" "$SIZE" 
