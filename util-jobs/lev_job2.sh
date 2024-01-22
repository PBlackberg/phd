#!/bin/bash
#SBATCH -J myjob           # Specify job name
#SBATCH -p shared          # Use partition shared
#SBATCH -N 1               # Specify number of nodes (1 for serial applications!)
#SBATCH -n 1               # Specify max. number of tasks to be invoked
#SBATCH -t 01:00:00        # Set a limit on the total run time
#SBATCH -A xz0123          # Charge resources on this project account
#SBATCH -o myjob.o%j       # File name for standard and error output

set -e

module load python3

echo "Start python script execution at $(date)"

python -u /path/to/myscipt.py