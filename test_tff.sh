#!/bin/bash

#SBATCH --job-name=tff_partition
#SBATCH --nodes=4
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=8g
#SBATCH --time=48:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard
#SBATCH --mail-user=apawlik@umich.edu
#SBATCH --mail-type=END
#SBATCH --output=results/tff.%A.%a.out

#SBATCH --array=1-8

module load python3.7-anaconda
module load cudnn/10.0-v7.6
module load cuda/10.1.105
module list

mkdir results/${SLURM_ARRAY_JOB_ID}
mkdir results/${SLURM_ARRAY_JOB_ID}/log

python tff_main.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID

# move output file to /log/ subfolder, only CSV files remain in main batch folder
mv results/tff.${SLURM_ARRAY_JOB_ID}.${SLURM_ARRAY_TASK_ID}.out results/${SLURM_ARRAY_JOB_ID}/log/
