
# sbatch $1
echo  $(squeue --me)

    # Get the job ID for script_1
JOB_ID=$(squeue --me | grep 'script_1' | awk '{print $1}')

    # Wait a moment for the job to start
sleep 2

# Tail the output file
echo "slurm-${JOB_ID}.out"
tail -f slurm-${JOB_ID}.out