#!/bin/bash
#SBATCH --job-name=training_job
#SBATCH --partition=gpu
#SBATCH --account=sscm033324
#SBATCH --gres=gpu:1
#SBATCH --mem=80GB
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=training_%j.out
#SBATCH --error=training_%j.err

# Load modules or activate conda env if needed
#module load cuda/11.8 
#module purge
#module load cuda/12.4.0

python U4_UnetTraining.py 
