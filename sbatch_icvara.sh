#!/bin/bash

#SBATCH --time 00-00:40:00
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 30



env | grep SLURM

module load gcc python

DIR=/users/ibarbier
source $DIR/myfirstvenv/bin/activate


python3 abc_smc.py 

python3 plot.py 

deactivate
