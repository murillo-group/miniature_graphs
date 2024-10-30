#!/usr/bin/bash
#SBATCH --job-name=paramsRoadNET
#SBATCH --output=out_%x/output_%A_%a.out
#SBATCH --error=out_%x/error_%A_%a.err
#SBATCH --array=1-9
#SBATCH --ntasks=1
#SBATCH --time=23:59:00  # Time limit
#SBATCH --mem=10G

# Source profile
source ~/.profile_minigraphs

# Calculate miniaturization size
FRAC=$(($SLURM_ARRAY_TASK_ID * 1))

bash params.sh "giant_inf-roadNET-PA" "$FRAC" 25 10 25




