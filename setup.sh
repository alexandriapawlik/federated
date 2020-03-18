#!/bin/bash

#SBATCH --job-name=tf_test
#SBATCH --account=tewaria1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=5gb
#SBATCH --mail-type=FAIL

# Load modules
module load python3.6-anaconda
module load cuda/10.0.130 cudnn/10.0-v7.6
module list

# Run the test
python3 /sw/examples/tensorflow/tf.py