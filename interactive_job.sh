#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=10GB 
#PBS -q normal         
#PBS -P k10
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/k10+gdata/hh5      
#PBS -l wd
#PBS -M <philip.blackberg@monash.edu> 
#PBS -I

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
python /g/data/k10/cb4968/phd/test.py $PBS_NCPUS 


# submitting job: 
# qsub /g/data/k10/cb4968/phd/interactive_job.sh 

# in ipython, run scripts from: 
# cd /g/data/k10/cb4968/phd/


# check status: 
# qstat -swx jobID (or qstat -s)

# check utilisation rate (%gpu): 
# nqstat_anu jobID

# remove job from queue: 
# qdel jobID  
