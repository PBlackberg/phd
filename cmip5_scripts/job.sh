# submitting job: qsub job.sh 
# check status: qstat -swx jobID
# check utilisation rate (%gpu): nqstat_anu jobID
# remove job from queue: qdel jobID 

!/bin/bash

PBS -l ncpus = 1
PBS -l mem = 190 GB
PBS -l jobfs = 200 GB # paralellisation (local disk on a compute node)
PBS -q normal         # express, normal, copyq(internet access), hugemem, megamem, gpuvolta(volta gpu) 
PBS -P k10
PBS -l walltime = 2:00:00
PBS -l storage = gdata/al33 + gdata/k10+scratch/k10 + gdata/hh5 # data: /g/data/al33/replicas/CMIP5/combined/
                                                                # scipts run from: /g/data/k10/cb4968/phd/cmip6_scripts
                                                                # anaconda environemt: /g/data/hh5/public/modules/conda/analysis3-unstable             
PBS -l wd
PBS -M <philip.blackberg@monash.edu> 
PBS -m <abe>	

module load python3/11

python3 super.py $PBS_NCPUS > /g/data/k10/cb4968/job_logs/$PBS_JOBID.log







