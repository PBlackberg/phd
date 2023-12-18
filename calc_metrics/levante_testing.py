
# -------------------
# loading environment
# -------------------
# ssh.config: file for setting initiation: module avail, module load python3/unstable, module load git/2.31.1-gcc-11.2.0, module list
# spack:      used as package manager to build and maintain the software tree (https://spack.readthedocs.io)
# To use module in script:  in bash or ksh script source /sw/etc/profile.levante
#                           in tcsh or csh script source /sw/etc/csh.levante


# in tcsh or csh script
source /sw/etc/csh.levante

import numpy as np
import xarray as xr
import os
# import cartopy
# import xesmf as xe

#asdkadskavn

# -------------------
#   Loading data
# -------------------
# DYAMOND data library at:   /work/bk1040/DYAMOND/ (most of the data is archived in the DKRZ tape system)
# DYAMOND summer and winter: /work/bk1040/DYAMOND/data/
# post processed data:       /work/bk1040/DYAMOND/data/winter_data/DYAMOND_WINTER/


# Winter simulation
def concat_files(path_folder, experiment):
    ''' Concatenates files of monthly or daily data between specified years
    (takes out a little bit wider range to not exclude data when interpolating grid) '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    year1, year2 = (1970, 1999)                      # range of dates to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) # indicates between which characters in the filename the first date is described (count starting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2)  # where the last date is described
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    # print(paths[0])
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35)) # take out a little bit wider range to not exclude data when interpolating grid
    return ds

folder = '/work/bk1040/DYAMOND/data/winter_data/DYAMOND_WINTER/MPIM-DWD-DKRZ/ICON-SAP-5km/DW-ATM/atmos/6hr/pr/dpp0014/ml/gn'
filename = 'pr_6hr_ICON-SAP-5km_DW-ATM_dpp0014_ml_gn_20200120000000-20200120180000.nc'
a = xr.open_dataset(f'{folder}/{filename}')
da = a['pr']
print(a)
print(da)
print(da.time.data)



# -------------------
#    Saving data
# -------------------
# store scripts at:     /work/bb1153/<user-account>/ (30 GiB) 
# store data at:        /work/bb1153
# scratch directory:    /scratch/b/b382628 (15 TiB automatically deleted every 14 days)
# swift object storage: account code: bb1153, https://swiftbrowser.dkrz.de/
























