#!/bin/bash
#SBATCH --job-name=glitch_fitter
#SBATCH --output=logs/glitch_fitter_%A_%a.out
#SBATCH --error=logs/glitch_fitter_%A_%a.err
#SBATCH --array=0-9
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G

module load  python-scientific/3.10.4-foss-2022a
source /fred/oz303/avajpeyi/venvs/glitch/bin/activate
python run_glitches.py ${SLURM_ARRAY_TASK_ID}
