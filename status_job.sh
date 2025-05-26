#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avinashnandyala921@gmail.com
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:02:00

squeue --me
tail -n 10 -f slurm-34127015.out

