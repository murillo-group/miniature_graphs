#!/bin/bash
#SBATCH --job-name=simTVshow
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00

source ~/.profile_minigraphs

MODEL="sir"
GRAPH_NAME="giant_soc-pages-tvshow"
N_ITERATIONS=1000

FILE_NAME="$DATA_DIR/networks/$GRAPH_NAME/graph.npz"
FILE_NAME_OUT="$DATA_DIR/networks/$GRAPH_NAME/simulations/original"

for i in {1..30}
do
    python -u "../simulations/$MODEL.py" "$FILE_NAME" "$N_ITERATIONS" "$FILE_NAME_OUT"
done
