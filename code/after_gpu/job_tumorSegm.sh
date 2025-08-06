#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:0:0
#SBATCH --mem=2G
#SBATCH --account=sscm033324


#nvidia-smi
module add languages/python/tensorflow-2.16.1
#pip install segmentation-models-3D

#python preprocessing.py
#python image_preparation.py
python tumorSegm.py

