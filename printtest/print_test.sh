#!/bin/bash

#SBATCH --job-name=tff_partition_test
#SBATCH --nodes=4
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5g
#SBATCH --time=48:00:00
#SBATCH --account=tewaria1
#SBATCH --partition=standard
#SBATCH --mail-user=apawlik@umich.edu
#SBATCH --mail-type=END
#SBATCH --print.log

module load python3.7-anaconda
python print_test.py