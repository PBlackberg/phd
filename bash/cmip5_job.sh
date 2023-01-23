#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=25GB 
#PBS -q normal         
#PBS -P k10
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/ua8+gdata/al33+gdata/k10+gdata/hh5         
#PBS -l wd
#PBS -M <philip.blackberg@monash.edu> 


module use /g/data/hh5/public/modules
module load conda/analysis3-22.10-unstable
python /g/data/k10/cb4968/phd/clVars.py $PBS_NCPUS 




# submitting job: qsub /g/data/k10/cb4968/phd/bash/cmip5_job.sh 



# check status: qstat -swx jobID (or qstat -s)
# check utilisation rate (%gpu): nqstat_anu jobID
# remove job from queue: qdel jobID 



# run from cd /g/data/k10/cb4968/phd/cmip5_scripts/cmip5_metrics/funcs/vars
# -I (for interactive job)
# -m abe	
# for cmip5 project al33
