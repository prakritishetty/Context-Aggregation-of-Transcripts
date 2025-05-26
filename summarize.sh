#!/bin/bash
#SBATCH -p gpu-preempt  # Partition
#SBATCH -t 01:00:00  # Job time limit (1 hour should be enough for combining files)
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -e slurm-%j-error.out
#SBATCH --gpus=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

python3 -m venv myenv
source myenv/bin/activate

echo "Installing requirements"
# pip install -r requirements.txt
echo "Finished installing requirements"

echo "Running summarize.py to generate summaries"
python summarize.py

echo "End time: $(date)"
