## Environment

This code was run on a Huawei Matebook 13 (Windows 11), on High Performance Computer (HPC)/ bluecrystal

The environment for this project can be replicated using: module add languages/python/tensorflow-2.16.1
and pip installing segmentation-models-3D

## Description of the data

To obtain the data refer to directory /user/home/ms13525/scratch/mshds-ml-data-2025 

## Pipeline

The pipeline for this project can all be found in tumourSeg.sh script

Pipline is 5 scripts

1_preprocessing.py  > Extracts data from source and organise into patient folders

2_Converting_to_Numpy.py >  Converting images into needed for segmentation

Training_Models.py  > Iterates through training models on different combination of modalities

4_Test_Evaluation.py  > Prints out evaluation Metrics results

5_Visualisepred.py > Visualise prediction for some patients

Full descriptions each script can be found at the head. 

# Running sbatch tumourSeg.sh will iterate between each script on order.

To run the pipeline on HPC make sure you have the correct resources to successfully run this job script. My template 
looks like the following:

```yaml
default-resources:

#SBATCH --job-name=test_job
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --time=1:0:0
#SBATCH --mem=2G
#SBATCH --account=sscm033324

