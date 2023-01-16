import xarray as xr
import os
import numpy as np
import matplotlib.pyplot as plt
import xesmf as xe


path_gen = '/g/data/ua8/Precipitation/GPCP/day/v1-3'
years = range(1996,2023)
folders = [f for f in os.listdir(path_gen) if (int(f) in years)]
folders = sorted(folders, key=int)



def regrid_conserv(ds_in, haveDsOut, path='/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc', modelDsOut='FGOALS-g2'):

    if haveDsOut:
        ds_out = xr.open_dataset(path)
        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
    
    else:
        pass
        
    return regrid(ds_in)

    

ds_list = []
for folder in folders:
    path_folder = os.path.join(path_gen, folder)
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))


    ds_in = xr.open_mfdataset(path_fileList, combine='by_coords')
    haveDsOut = True
    ds_year = regrid_conserv(ds_in, haveDsOut)

    ds_list.append(ds_year)


ds_concat = xr.concat(ds_list, dim='time')

















