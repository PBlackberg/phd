#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=20GB 
#PBS -q normal         
#PBS -P k10
#PBS -l walltime=10:00:00
#PBS -l storage=gdata/al33+gdata/k10+gdata/hh5         
#PBS -l wd
#PBS -M <philip.blackberg@monash.edu> 
#PBS -m abe	

module use /g/data/hh5/public/modules
module load analysis3-unstable
python cmip5_scripts/cmip5_metrics/cmip5_metrics.py $PBS_NCPUS 




# other specifications
# not sure what this means: +scratch/k10
# how large to use hugemem or megamem ?
##PBS -l jobfs=20GB ## parallellisation (local disk on a compute node)


# directories to use
# data: /g/data/al33/replicas/CMIP5/combined/
# scipts run from: /g/data/k10/cb4968/phd/cmip5_scripts
# anaconda environemt: /g/data/hh5/public/modules/conda/analysis3-unstable    $

# general
# the bash line needs to be on line 1
# remove spaces between + and =

# Queues
# express, normal, copyq(internet access), hugemem, megamem, gpuvolta(volta gpu) 

# submitting
# submitting job: qsub cmip5_scripts/bash/cmip5_job.sh 
# check status: qstat -swx jobID (or qstat -s)
# check utilisation rate (%gpu): nqstat_anu jobID
# remove job from queue: qdel jobID 


# > /g/data/k10/cb4968/phd/cmip5_scripts/bash/$PBS_JOBID.log