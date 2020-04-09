#!/bin/bash

#SBATCH --job-name=tff_partition
#SBATCH --nodes=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5g
#SBATCH --time=48:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard
#SBATCH --mail-user=apawlik@umich.edu
#SBATCH --mail-type=END

#SBATCH --array=1-14

module load python3.7-anaconda
module load cudnn/10.0-v7.6
module load cuda/10.1.105
module list

mkdir results/${SLURM_ARRAY_JOB_ID}

python tff_main.py $SLURM_ARRAY_TASK_ID > results/${SLURM_ARRAY_JOB_ID}/tff.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}.log
