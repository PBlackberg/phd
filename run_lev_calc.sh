#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --partition=compute
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=128
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --mail-type=FAIL
#SBATCH --account=xz0123
#SBATCH --output=my_job.%j.out

# limit stacksize ... adjust to your programs need
# and core file size
ulimit -s 204800
ulimit -c 0

# Replace this block according to https://docs.dkrz.de/doc/levante/running-jobs/runtime-settings.html#mpi-runtime-settings
echo "Replace this block according to  https://docs.dkrz.de/doc/levante/running-jobs/runtime-settings.html#mpi-runtime-settings"
exit 23
# End of block to replace

# Use srun (not mpirun or mpiexec) command to launch
# programs compiled with any MPI library
srun -l --cpu_bind=verbose --hint=nomultithread \
  --distribution=block:cyclic ./myprog