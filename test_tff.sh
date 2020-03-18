#!/bin/bash

#SBATCH --job-name tff_main
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3g
#SBATCH --time=40:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard
#SBATCH --mail-user=apawlik@umich.edu
#SBATCH --mail-type=END
#SBATCH --output=results/tff.%A.%a.log

#SBATCH --array=1-2

module load python3.7-anaconda
module load cuda/10.0.130 cudnn/10.0-v7.6
module list

mkdir results/${SLURM_ARRAY_JOB_ID}

python tff_main.py $SLURM_ARRAY_TASK_ID

mv results/tff.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}.log results/${SLURM_ARRAY_JOB_ID}
