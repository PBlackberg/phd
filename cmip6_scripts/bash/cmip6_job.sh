#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=50GB 
#PBS -q normal         
#PBS -P k10
#PBS -l walltime=00:20:00
#PBS -l storage=gdata/oi10+gdata/k10+gdata/hh5         
#PBS -l wd
#PBS -M <philip.blackberg@monash.edu> 
#PBS -m abe	

module use /g/data/hh5/public/modules
module load conda/analysis3-22.10
python cmip6_scripts/cmip6_metrics/funcs/vars/mseVars.py $PBS_NCPUS 


# submitting job: qsub cmip6_scripts/bash/cmip6_job.sh 

# check status: qstat -swx jobID (or qstat -s)
# check utilisation rate (%gpu): nqstat_anu jobID
# remove job from queue: qdel jobID  






