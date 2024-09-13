#!/bin/bash
#SBATCH --job-name=glitch_fitter
#SBATCH --output=logs/glitch_fitter_%A_%a.out
#SBATCH --error=logs/glitch_fitter_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

# Load necessary modules
module load python/3.8

# Run the Python script with the array index as an argument
python studies/run_glitches.py ${SLURM_ARRAY_TASK_ID}
