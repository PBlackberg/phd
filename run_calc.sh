#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=50GB 
#PBS -q normal         
#PBS -P w40
#PBS -l walltime=00:30:00
#PBS -l storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/k10+gdata/hh5+gdata/rt52
#PBS -l wd
#PBS -m abe
#PBS -M <philip.blackberg@monash.edu> 
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
python /home/565/cb4968/Documents/code/phd/switch/calculation_large_scale_env.py $PBS_NCPUS 



# submitting job: 
# qsub /home/565/cb4968/Documents/code/phd/run_calc.sh 

# in ipython, run scripts from: 
# cd /g/data/k10/cb4968/phd/metrics/funcs/vars

# check status: 
# qstat -swx jobID (or qstat -s)

# check utilisation rate (%gpu): 
# nqstat_anu jobID

# remove job from queue: 
# qdel jobID  


# interactive job
# qsub -I -normal  -Pk10 -lwalltime=01:00:00,ncpus=1,mem=15GB,jobfs=200GB,storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/k10+gdata/hh5,wd


