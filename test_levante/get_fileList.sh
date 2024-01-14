# define variables 
varname='rlut'
expid='ngc2013'
catalog=/home/k/k203123/NextGEMS_Cycle2.git/experiments/${expid}/scripts/${expid}_new.json

# Obtain a list of path and file names that contain specified variable and frequency
#--frequency=3hour 
varfilelist=`find_files --catalog_file=${catalog} ${varname} ${expid} --frequency=3hour --time_range 2020-02-01 2021-02-01T23`

# Print the file paths
echo "$varfilelist"









