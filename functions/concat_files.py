import xarray as xr
import numpy as np
import os


def latestVersion(path):
    ''' Picks the latest version if there are multiple '''
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def concat_files(path_gen, variable, model, experiment):
    ''' Concatenates files of monthly or daily data between specified years
    (takes out a little bit wider range to not exclude data when interpolating grid) '''
    path_folder =  os.path.join(path_gen, grid_folder(model), latestVersion(os.path.join(path_gen, grid_folder(model, experiment, variable))))
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    
    year1, year2 = (1970, 1999) if experiment == 'historical' else (2070, 2099) # range of years to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder else (17, 13) # indicates between which characters in the filename the first fileyear is described (count starting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2) if 'Amon' in path_folder else (8, 4) # where the last fileyear is described

    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart,:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd] <= year2 and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]

    # for f in files:  # one model from warming scenario from cmip5 have a file that needs to be removed (creates duplicate data otherwise)
    #     files.remove(f) if f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]=='19790101' and f[f.index(".nc")-fileYear2_charStart : f.index(".nc")]=='20051231' else None
            
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35)) # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# ---------------------------------------------------------------------------------------- for cmip6 data ----------------------------------------------------------------------------------------------------------#
def choose_cmip6_ensemble(model):
    ''' Some models don't have the ensemble most common amongst other models 
    and some experiments don't have the same ensemble as the historical simulation'''
    ensemble = 'r1i1p1f2' if model in ['CNRM-CM6-1', 'UKESM1-0-LL'] else 'r1i1p1f1'
    return ensemble

def grid_folder(model):
    ''' Some models have a different grid folder in the path to the files'''
    folder = 'gn'
    folder = 'gr' if model == 'CNRM-CM6-1' else None
    folder = 'gr1' if model == 'GFDL-CM4' else None           
    return folder
    
# ----------------------------------------------------------------------------------------- for cmip5 data ----------------------------------------------------------------------------------------------------------#
def choose_cmip5_ensemble(model, experiment):
    ''' Some models don't have the ensemble most common amongst other models 
    and some experiments don't have the same ensemble as the historical simulation'''
    ensemble = 'r6i1p1' if model in ['EC-EARTH', 'CCSM4'] else 'r1i1p1'

    ensemble = 'r6i1p1' if model == 'GISS-E2-H' and experiment == 'historical' else 'r1i1p1'
    ensemble = 'r2i1p1' if model == 'GISS-E2-H' and not experiment == 'historical' else 'r1i1p1'
    return ensemble













































































