#!/bin/bash
#SBATCH -p gpu-preempt  # Partition
#SBATCH -t 08:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -e slurm-%j-error.out
#SBATCH --gpus=1

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

python3 -m venv myenv
source myenv/bin/activate

echo "Installing requirements"
pip install -r requirements_eval.txt
echo "Finished installing requirements"
export CUDA_LAUNCH_BLOCKING=1
# python main_2.py
python main_evaluation.py