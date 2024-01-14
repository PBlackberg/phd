#!/bin/bash
#SBATCH --job-name=testJob # This sets the job name
#SBATCH --ntasks=4         # This sets the number of tasks

# The following line is a comment and will be ignored by SLURM and bash.
echo "Starting job"

# The SLURM scheduler interprets the #SBATCH directives,
# but bash will run the echo command.



# ----------------
# run slurm script
# -----------------
# sbatch slurm_script.sh
# This needs SLURM installed which is mainly a HPC module



