#!/bin/bash
#SBATCH --job-name=tf_gpu_deep_learning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks=1                    # Number of tasks (usually 1 for TensorFlow)
#SBATCH --mem=100G                    # Memory requested (adjust if needed)
#SBATCH --time=2-00:00:00             # Max runtime: 2 days (can go up to 14 days max)
#SBATCH --output=tf_cgu_%j.out        # Stdout file
#SBATCH --error=tf_cgu_%j.err         # Stderr file
#SBATCH --account=sscm033324          # Your account name


module load languages/python/tensorflow-2.16.1

#module load cuda/12.4.0-z7k5
#module load cudnn/8.9.7.29-12-dyi6

#python 1_moving_folders.py
#python 2_getting_data.py
python U4_UnetTraining.py 
