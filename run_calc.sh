#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=50GB 
#PBS -q normal         
#PBS -P w40
#PBS -l walltime=05:00:00
#PBS -l storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/k10+gdata/hh5+gdata/rt52
#PBS -l wd
#PBS -m abe
#PBS -M <philip.blackberg@monash.edu> 
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
python /home/565/cb4968/Documents/code/phd/switch/calculate_ls_env.py $PBS_NCPUS 



# -------------
# submit script
# -------------
# qsub /home/565/cb4968/Documents/code/phd/run_calc.sh    # submit
# qstat -swx jobID (or qstat -s)                          # check status
# nqstat_anu jobID                                        # check utilisation rate (%gpu) 
# qdel jobID                                              # remove from queue


# --------------------
# interactive terminal
# --------------------
# qsub -I -qnormal  -Pw40 -lwalltime=01:00:00,ncpus=1,mem=50GB,jobfs=200GB,storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/rt52+gdata/k10+gdata/hh5,wd
# module use /g/data/hh5/public/modules
# module load conda/analysis3-unstable



