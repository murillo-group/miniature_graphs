#!/usr/bin/bash
#SBATCH --job-name=minHamsterster
#SBATCH --output=out_%x/output_%A_%a.out
#SBATCH --error=out_%x/error_%A_%a.err
#SBATCH --array=1-9
#SBATCH --ntasks=6
#SBATCH --time=5:00:00  # Time limit
#SBATCH --mem=7G

# Source profile
source ~/.profile_minigraphs

# Calculate miniaturization size
FRAC=$(($SLURM_ARRAY_TASK_ID * 100))

bash miniaturize.sh "giant_hamsterster" "$FRAC" 10 20000 200




