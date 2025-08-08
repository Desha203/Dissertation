#!/bin/bash

#SBATCH --job-name=test_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=3:0:0
#SBATCH --mem=2G
#SBATCH --account=sscm033324


#nvidia-smi
module add languages/python/tensorflow-2.16.1
pip install segmentation-models-3D

#python 1_preprocessing.py  
#python 2_Converting_to_Numpy.py  
#python Training_Models.py
python 4_Evaluation_on_Test.py    


