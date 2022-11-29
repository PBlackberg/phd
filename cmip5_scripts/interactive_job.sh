

!/bin/bash

PBS -l ncpus = 1
PBS -l mem = 190 GB
PBS -l jobfs = 200 GB
PBS -q normal
PBS -P k10
PBS -l walltime = 2:00:00
PBS -l storage = gdata/al33 + gdata/k10+scratch/k10 + gdata/hh5 # data: /g/data/al33/replicas/CMIP5/combined/ , scipts ran from: /g/data/k10/cb4968/phd/cmip5_scripts
PBS -l wd
PBS -M <philip.blackberg@monash.edu> 
PBS -m <abe>	

module load python3/11

python3 super.py $PBS_NCPUS > /g/data/k10/cb4968/job_logs/$PBS_JOBID.log







