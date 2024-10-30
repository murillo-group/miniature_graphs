#!/usr/bin/bash
#SBATCH --job-name=simTVshow
#SBATCH --output=out_%x/output_%A_%a.out
#SBATCH --error=out_%x/error_%A_%a.err
#SBATCH --array=0-4
#SBATCH --ntasks=1
#SBATCH --time=00:15:00  # Time limit
#SBATCH --mem=3G

# Source profile
source ~/.profile_minigraphs

DIRS=(200_2024-10-13_00  500_2024-10-14_00
300_2024-10-20_00  600_2024-10-14_00
400_2024-10-13_00)

DIR=${DIRS[$SLURM_ARRAY_TASK_ID]}

MODEL="sir"
GRAPH_NAME="giant_soc-pages-tvshow"
N_ITERATIONS=1000

FILE_NAME_IN="$DATA_DIR/networks/$GRAPH_NAME/miniatures/$DIR/graph.npz"
FILE_NAME_OUT="$DATA_DIR/networks/$GRAPH_NAME/simulations/$MODEL/$DIR/"

echo $FILE_NAME_OUT

for i in {1..30}
do
    python -u "../simulations/$MODEL.py" "$FILE_NAME_IN" "$N_ITERATIONS" "$FILE_NAME_OUT"
done




