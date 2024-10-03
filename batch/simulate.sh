#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=02:00:00

source ~/.profile_minigraphs

MODEL=$1
GRAPH_NAME=$2
N_ITERATIONS=$3

FILE_NAME="$DATA_DIR/networks/$GRAPH_NAME/graph.npz"
FILE_NAME_OUT="$DATA_DIR/simulations/$GRAPH_NAME/original"

for i in {1..30}
do
    python -u ../simulations/$MODEL.py "$FILE_NAME" "$N_ITERATIONS" "$FILE_NAME_OUT"
done
