.. _multidecadal:

Dealing with multidecadal 10km coupled simulations
--------------------------------------------------

Due to the output file structure (very complicated  but there are reasons why this is so), it would help to use intake to access/extract the data. One can work fully on python to do this, or an alternative is to use a "magic" script that Florian created to produce a file list that matches specific variable name and output frequency. cdo can then concatenate the files and extract the desired variable into one file. The alternative method is used here and it is all done on the shell environment. 

"Magic" script called :code:`find_files`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One can copy and paste :code:`find_files` content from `here <https://easy.gems.dkrz.de/Processing/Intake/find_files.html>`__


Extract a variable with a particular output frequency for a given period
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using the example of ngc2013::

  varname='rlut'
  expid='ngc2013'
  #expid='rthk001'
  catalog=/home/k/k203123/NextGEMS_Cycle2.git/experiments/${expid}/scripts/${expid}.json
  
  #Obtain a list of path and file names that contain specified variable and frequency
  varfilelist=`find_files --catalog_file=${catalog} ${varname} ${expid} --frequency=3hour `
  
  freqm='3hr'
  varfile=${expid}_${varname}_${freqm}.nc
  
  #Extract 3-hourly
  cdo -L -select,name=${varname} [ ${varfilelist} ] ${varfile}
  
  #Extract for a given time range
  varfilelist=`find_files --catalog_file=${catalog} ${varname} ${expid} --frequency=3hour --time_range 2020-02-01 2020-02-03T23 `
  cdo -L -select,name=${varname} [ ${varfilelist} ] ${expid}_${varname}_${freqm}_20200201-20200203.nc

Selection of variable over a region and take the area mean
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**atmosphere grid**::

   cdo -L -fldmean -sellatlonbox,-10,10,190,270 -setgrid,/pool/data/ICON/grids/public/mpim/0034/icon_grid_0033_R02B08_G.nc ${varfile} ${expid}_${varname}_${freqm}_boxmean_10S10N190E270E.nc

**ocean grid**::

   cdo -L -fldmean -sellatlonbox,-10,10,190,270 -setgrid,/pool/data/ICON/grids/public/mpim/0034/icon_grid_0034_R02B08_O.nc ${varfile} ${expid}_${varname}_${freqm}_boxmean_10S10N190E270E.nc

Remap weights for r2b8
~~~~~~~~~~~~~~~~~~~~~~
**ocean grid**::

   cdo -L -P 48 -gennn,r360x180 -setgrid,/pool/data/ICON/grids/public/mpim/0034/icon_grid_0034_R02B08_O.nc -seldate,2020-01-21 /work/mh0287/m300466/DPP/ngc2013/SST/SST_dd_20200121-20351231.nc /work/mh0256/m300466/DPP/weights/r2b8O_r360x180_nnremapweights.nc

**atm grid**::

   cdo -L -P 48 -gennn,r360x180 -setgrid,/pool/data/ICON/grids/public/mpim/0033/icon_grid_0033_R02B08_G.nc -selname,tas /work/bm1235/k203123/experiments/ngc2013/run_20200120T000000-20200131T235830/ngc2013_atm_2d_1mth_mean_20200120T000000Z.nc /work/mh0256/m300466/DPP/weight


Direct remapping onto 1deg x 1deg
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**sea level pressure (on atm grid)**
vi extract2datm_mm_ngc2013.sh::

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

Execute the script::

   ./extract2datm_mm_ngc2013.sh


**upper 500m ocean temperature**
vi extract3docn_mm_ngc2013.sh::

   #!/bin/sh

   find_files=/home/m/m300466/pyfuncs/find_files

   varname=to
   expid=ngc2013

   catalog=/home/k/k203123/NextGEMS_Cycle2.git/experiments/${expid}/scripts/${expid}.json
   freqm=mm
   ocngrid=/pool/data/ICON/grids/public/mpim/0034/icon_grid_0034_R02B08_O.nc
   remapwgt=/work/mh0256/m300466/DPP/weights/r2b8O_r360x180_nnremapweights.nc

   #Obtain a list of path and file names that contain specified variable, frequency and time range (Feb 2020 - Dec 2049)
   varfilelist=$( $find_files --catalog_file=${catalog} ${varname} ${expid} --frequency=1month --time_range 2020-03-01 2050-01-01T23)

   #Extract upper 500m, monthly mean and remap
   mkdir -p /work/mh0287/m300466/topaz/${expid}/${varname}
   varfile=/work/mh0287/m300466/topaz/${expid}/${varname}/${expid}_upper500m_${varname}_${freqm}_202002-204912_r360x180.nc

   cdo -L -P 48 -remap,r360x180,${remapwgt} -shifttime,-1day -setgrid,${ocngrid} -select,levidx=1/59,name=${varname} [ ${varfilelist} ] ${varfile}
   #We shift the time stamp by -1day in order to get it to the correct month.

Execute the script::

   ./extract3docn_mm_ngc2013.sh



.. note::

   See also the `CDO Wiki <https://code.mpimet.mpg.de/projects/cdo/wiki>`__ with `Tutorial <https://code.mpimet.mpg.de/projects/cdo/wiki/Tutorial>`__ and `FAQ <https://code.mpimet.mpg.de/projects/cdo/wiki/FAQ>`__
   and the `DKRZ pages on data processing tools <https://www.dkrz.de/up/services/analysis/data-processing/tools>`__, with additional  `examples <https://www.dkrz.de/up/services/analysis/data-processing/tools/cdo-examples>`__
