
''' 
DYAMOND data at: /work/bk1040/DYAMOND/ 
post-processed data sets under /work/bk1040/DYAMOND/data/winter_data/DYAMOND_WINTER/
To access further data files, you will need to make a request within the DYAMOND data library:  README file at /work/bk1040/DYAMOND/README

For grid converison:
For several models, the grid information is not included in the output. Instead it can be found in a grid.nc file. You can find this grid file in the frequency directory fx/ of the directory structure of the data, e.g. /work/bk1040/DYAMOND/data/winter_data/DYAMOND_WINTER/NOAA/SHiELD-3km/DW-ATM/atmos/fx/gn/grid.nc for the SHiELD data.
To associate data with the grid information, cdo will need a -setgrid,GRIDFILENAME, e.g.
cdo -sellonlatbox,0,20,40,60 -setgrid,GRIDFILENAME INFILE OUTFILE


''' 
















































