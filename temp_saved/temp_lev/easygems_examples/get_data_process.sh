#!/bin/sh
find_files=/home/m/m300466/pyfuncs/find_files
varname=psl
expid=ngc2013

catalog=/home/k/k203123/NextGEMS_Cycle2.git/experiments/${expid}/scripts/${expid}.json
freqm=mm
atmgrid=/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc

#Obtain a list of path and file names that contain specified variable, frequency and time range (Feb 2020 - Jan 2050)
varfilelist=$( $find_files --catalog_file=${catalog} ${varname} ${expid} --frequency=1month --time_range 2020-03-01 2050-02-01T23)

#Extract monthly mean and remap
mkdir -p /work/mh0287/m300466/topaz/${expid}/${varname}
varfile=/work/mh0287/m300466/topaz/${expid}/${varname}/${expid}_${varname}_${freqm}_202002-205001_r360x180.nc

cdo -L -P 48 -remapnn,r360x180 -shifttime,-1day -setgrid,${atmgrid} -select,name=${varname} [ ${varfilelist} ]  ${varfile}
# We shift the time stamp by -1day in order to get it to the correct month.




