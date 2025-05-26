#!/bin/bash

# Create a temporary job script
cat > status_job.sh << 'EOF'
#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=avinashnandyala921@gmail.com
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=00:02:00

squeue --me
tail -n 10 -f slurm-34127015.out

EOF

# Make the job script executable
chmod +x status_job.sh

while true; do
    # Submit the job
    sbatch status_job.sh
    
    # Wait for 2 minutes
    sleep 120

    # Monitor the last 10 lines of the SLURM output file
    
done
