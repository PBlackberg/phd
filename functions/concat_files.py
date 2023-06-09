import xarray as xr
import numpy as np
import os




# ---------------------------------------------------------------------------------------- for cmip6 data ----------------------------------------------------------------------------------------------------------#
def choose_ensemble(model, experiment):
    ''' Some models have different ensembles '''
    ensemble = 'r1i1p1f1'
    if model == 'CNRM-CM6-1' or model =='UKESM1-0-LL':
        ensemble = 'r1i1p1f2'
        
    return ensemble


def grid_label(model, experiment, variable):
    ''' Some models have a different grid folder in path to files'''
    folder = 'gn'
    if model == 'CNRM-CM6-1':
        folder = 'gr'
    if model == 'GFDL-CM4':
        folder = 'gr1'    
    return folder


def pick_latestVersion(path):
    ''' Picks the latest version if there are multiple '''
    versions = os.listdir(path)
    if len(versions)>1:
        version = max(versions, key=lambda x: int(x[1:]))
    else:
        version = versions[0]
    return version
    
    
def concat_files(path_gen, experiment, model, variable):
    ''' Concatenates files of monthly or daily data between specified years '''
    folder_grid = grid_label(model, experiment, variable)
    version = pick_latestVersion(os.path.join(path_gen, folder_grid))
    path_folder =  os.path.join(path_gen, folder_grid, version)
        
    if experiment == 'historical':
        yearEnd_first = 1970
        yearStart_last = 1999
        
    if experiment == 'ssp585':
        yearEnd_first = 2070
        yearStart_last = 2099

    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if 'Amon' in path_folder:
        files = sorted(files, key=lambda x: x[x.index(".nc")-13:x.index(".nc")-9])
        files = [f for f in files if int(f[f.index(".nc")-13:f.index(".nc")-9]) <= int(yearStart_last) and int(f[f.index(".nc")-6:f.index(".nc")-2]) >= int(yearEnd_first)]
    elif 'day' in path_folder:
        files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
        files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= int(yearStart_last) and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= int(yearEnd_first)]

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))
    ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35)) # take out a little bit wider range to not exclude data when interpolating grid
    return ds















