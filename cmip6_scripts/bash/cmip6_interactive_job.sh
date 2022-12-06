#!/bin/bash
#PBS -l ncpus=1
#PBS -l mem=20GB 
#PBS -q normal         
#PBS -P k10
#PBS -l walltime=24:00:00
#PBS -l storage=gdata/al33+gdata/k10+gdata/hh5         
#PBS -l wd
#PBS -M <philip.blackberg@monash.edu> 
#PBS -m abe	
#PBS -I
module load conda/analysis3-unstable
python3 cmip5_super.py $PBS_NCPUS > /g/data/k10/cb4968/phd/cmip6_scripts/$PBS_JOBID.log













