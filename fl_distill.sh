#!/bin/bash

#SBATCH --job-name fl_distill
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3g
#SBATCH --time=40:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --mail-user=apawlik@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=gl_results/lr_tune.%A_%a.log

#SBATCH --array=0-23

mkdir tf_results/${SLURM_ARRAY_JOB_ID}

python test.py $SLURM_ARRAY_TASK_ID