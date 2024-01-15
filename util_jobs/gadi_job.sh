#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=25GB 
#PBS -q normal         
#PBS -P w40
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/k10+gdata/hh5+gdata/rt52
#PBS -l wd
#PBS -m abe
#PBS -M <philip.blackberg@monash.edu> 
#PBS -j oe

module use /g/data/hh5/public/modules
module load conda/analysis3-unstable
python /home/565/cb4968/Documents/code/phd/switch/calculate_clouds.py $PBS_NCPUS 



# comment out from here if running bash script
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
# for packages
module use /g/data/hh5/public/modules
module load conda/analysis3-unstable

# for interactive job
qsub -I -qnormal  -Pw40 -lwalltime=8:00:00,ncpus=1,mem=50GB,jobfs=200GB,storage=gdata/al33+gdata/oi10+gdata/ia39+gdata/rt52+gdata/fs38+gdata/k10+gdata/hh5,wd
qstat

# for filetransfer
scp -r cb4968@gadi-dm.nci.org.au:/g/data/k10/cb4968/data/metrics /Users/cbla0002/Desktop/


# for debug plot
import os
print(da)
plot_object = da.isel(time=0).plot()
fig = plot_object.figure
fig.savefig(f'{os.getcwd()}/test/plot_test/test.png')
print(da.isel(time=0))



