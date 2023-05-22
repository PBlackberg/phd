import xarray as xr
import xesmf as xe
import numpy as np
import os
import scipy
home = '/g/data/k10/cb4968'


# --------------------------------------------------------- finding ensemble/version and concatenating files --------------------------------------------------------

def data_exist(model, experiment, variable):
    ''' Some models don't have all the data '''
    data_exist = True
    if variable == 'hus':
        if model == 'EC-EARTH' and experiment == 'rcp85':
            data_exist = False
    
    if variable == 'hur':
        if model == 'EC-EARTH':
            data_exist = False

    if variable == 'wap':
        if model == 'GISS-E2-H' or model == 'HadGEM2-CC' or model == 'EC-EARTH':
            data_exist = False

    if variable == 'cl':
        if model == 'CNRM-CM5' or model == 'CCSM4' or model == 'HadGEM2-AO' or model == 'EC-EARTH':
            data_exist = False
        if model == 'CESM1-BGC' and experiment == 'rcp85':
            data_exist = False
    return data_exist


def choose_ensemble(model, experiment):
    ''' Some models have different ensembles '''
    ensemble = 'r1i1p1'
    if model == 'GISS-E2-H' and experiment == 'historical':
        ensemble = 'r6i1p1'
    if model == 'GISS-E2-H' and experiment == 'rcp85':
        ensemble = 'r2i1p1'
    if model == 'EC-EARTH':
        ensemble = 'r6i1p1'
    if model == 'CCSM4':
        ensemble = 'r6i1p1'
    return ensemble


def pick_latestVersion(path_gen, ensemble):
    ''' Picks the latest version if there are multiple '''
    versions = os.listdir(os.path.join(path_gen, ensemble))
    if len(versions)>1:
        version = max(versions, key=lambda x: int(x[1:]))
    else:
        version = versions[0]
    return version


def concat_files(path_folder, experiment):
    ''' Concatenates files of monthly or daily data between specified years '''
    if experiment == 'historical':
        yearEnd_first = 1970
        yearStart_last = 1999

    if experiment == 'rcp85':
        yearEnd_first = 2070
        yearStart_last = 2099

    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if 'Amon' in path_folder:
        files = sorted(files, key=lambda x: x[x.index(".nc")-13:x.index(".nc")-9])
        files = [f for f in files if int(f[f.index(".nc")-13:f.index(".nc")-9]) <= yearStart_last and int(f[f.index(".nc")-6:f.index(".nc")-2]) >= yearEnd_first]
    else:
        files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
        files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= yearStart_last and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= yearEnd_first]
        for f in files:
            if int(f[f.index(".nc")-17:f.index(".nc")-9])==19790101 and int(f[f.index(".nc")-8:f.index(".nc")])==20051231: # one model have a file that needs to be removed (creates duplicate data otherwise)
                files.remove(f)

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))
    return ds


def regrid_conserv_xesmf(ds_in):
    ''' Creates regridder, interpolating to the grid of FGOALS-g2 '''
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr' # interpolate grid to grid from FGOALS-g2
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'

    ds_out = xr.open_dataset('{}/{}'.format(folder, fileName)).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)
    return regridder


# -------------------------------------------------------------------------------- get variable -------------------------------------------------------------------------------

def get_pr(institute, model, experiment, resolution):
    ''' Create path to files for precipitation data and concatenate to dataset (including regridding if necessary) '''
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    variable = 'pr'
    if not data_exist(model, experiment, variable):
        ds_pr = xr.Dataset(
            data_vars = {'precip': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment) # picks out lat: -35, 35

        precip = ds['pr']*60*60*24 # convert to mm/day
        precip.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9) # give new units
        
        if resolution == 'orig':
            ds_pr = xr.Dataset(
                data_vars = {'precip': precip},
                attrs = ds.attrs
                )
        elif resolution == 'regridded':
            regridder = regrid_conserv_xesmf(ds) # define regridder based of grid from other model
            precip_n = regridder(precip) # conservatively interpolate to grid from other model, onto lat: -30, 30 (_n is new grid)
            ds_pr = xr.Dataset(
                data_vars = {'precip': precip_n},
                attrs = ds.attrs
                )
    return ds_pr


def get_tas(institute, model, experiment, resolution):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    if model == 'FGOALS-g2':
        path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/mon/atmos/Amon'   
    variable = 'tas'

    if not data_exist(model, experiment, variable):
        ds_tas = xr.Dataset(
            data_vars = {'precip': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        tas = ds['tas']-273.15 # convert to degrees Celsius
        tas.attrs['units']= '\u00B0C'

        if resolution == 'orig':
            ds_tas = xr.Dataset(
                data_vars = {'tas': tas},
                attrs = ds.attrs
                )
        elif resolution == 'regridded':
            regridder = regrid_conserv_xesmf(ds)
            tas_n = regridder(tas)
            ds_tas = xr.Dataset(
                data_vars = {'tas': tas_n},
                attrs = ds.attrs
                )
    return ds_tas


def get_hus(institute, model, experiment, resolution):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    if model ==  'CESM1-BGC' or ((model == 'HadGEM2-AO' or model == 'EC-EARTH') and experiment == 'rcp85'):
        path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/mon/atmos/Amon'   
    variable = 'hus'
    if not data_exist(model,experiment, variable):
        ds_hus = xr.Dataset(
            data_vars = {'hus': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        hus = ds['hus'] # unitless kg/kg

        if resolution == 'orig':
            ds_hus = xr.Dataset(
                data_vars = {'hus': hus},
                attrs = ds.attrs
                )
        elif resolution == 'regridded':
            regridder = regrid_conserv_xesmf(ds)
            hus_n = regridder(hus)
            ds_hus = xr.Dataset(
                data_vars = {'hus': hus_n},
                attrs = ds.attrs
                )
    return ds_hus


def get_hur(institute, model, experiment, resolution):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    variable = 'hur'
    if not data_exist(model,experiment,variable):
        ds_hur = xr.Dataset(
            data_vars = {'hur': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        hur = ds['hur'] # units in %
        hur.attrs['units']= '%'

        if resolution == 'orig':
            ds_hur = xr.Dataset(
                data_vars = {'hur': hur},
                attrs = ds.attrs
                )
        elif resolution == 'regridded':
            regridder = regrid_conserv_xesmf(ds)
            hur_n = regridder(hur) 
            ds_hur = xr.Dataset(
                data_vars = {'hur': hur_n},
                attrs = ds.attrs
                )
    return ds_hur



def get_wap(institute, model, experiment, resolution):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    if model == 'CCSM4' or model == 'HadGEM2-AO' or model == 'inmcm4' or model =='CESM1-BGC' or (model == 'bcc-csm1-1' and experiment=='rcp85'):
        path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/mon/atmos/Amon'   
    
    variable = 'wap'
    if not data_exist(model,experiment,variable):
        ds_wap = xr.Dataset(
            data_vars = {'wap': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        wap = ds['wap']*60*60*24/100 # convert to hPa/day   
        wap.attrs['units']= 'hPa day' + chr(0x207B) + chr(0x00B9) 

        if resolution == 'orig':
            ds_wap = xr.Dataset(
                data_vars = {'wap': wap},
                attrs = ds.attrs
                )
        elif resolution == 'regridded':
            regridder = regrid_conserv_xesmf(ds)
            wap_n = regridder(wap)
            ds_wap = xr.Dataset(
                data_vars = {'wap': wap_n},
                attrs = ds.attrs
                )
    return ds_wap



def get_cl(institute, model, experiment, resolution):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    variable = 'cl'
    if not data_exist(model,experiment,variable):
        ds_cl = xr.Dataset(
            data_vars = {'cl': np.nan}
            )
        ds_p_hybridsigma = xr.Dataset(
            data_vars = {'p_hybridsigma': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        if model == 'IPSL-CM5A-MR' or model == 'MPI-ESM-MR' or model=='CanESM2': # different models have different conversions from height coordinate to pressure coordinate.
            p_hybridsigma = ds.ap + ds.b*ds.ps
        elif model == 'FGOALS-g2':
            p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
        elif model == 'HadGEM2-CC':
            p_hybridsigma = ds.lev+ds.b*ds.orog
        else:
            p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps
        
        if resolution == 'orig':
            ds_cl = ds # units in % on sigma pressure coordinates
            
            ds_p_hybridsigma = xr.Dataset(
                data_vars = {'p_hybridsigma': p_hybridsigma},
                attrs = ds.attrs
                )

        elif resolution == 'regridded':
            cl = ds['cl'] # units in % on sigma pressure coordinates
            
            regridder = regrid_conserv_xesmf(ds)
            cl_n = regridder(cl)
            p_hybridsigma_n = regridder(p_hybridsigma)

            ds_cl = xr.Dataset(
                data_vars = {'cl': cl_n},
                attrs = ds.attrs
                )
            ds_p_hybridsigma = xr.Dataset(
                data_vars = {'p_hybridsigma': p_hybridsigma_n},
                attrs = ds.attrs
                )
    return ds_cl, ds_p_hybridsigma





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import sys
    sys.path.insert(0, '{}/phd/functions'.format(home))
    from myFuncs import *
    
    
    models = [
        # 'IPSL-CM5A-MR', # 1
        # 'GFDL-CM3',     # 2
        # 'GISS-E2-H',    # 3
        'bcc-csm1-1',   # 4
        # 'CNRM-CM5',     # 5
        'CCSM4',        # 6
        'HadGEM2-AO',   # 7
        # 'BNU-ESM',      # 8
        # 'EC-EARTH',     # 9
        # 'FGOALS-g2',    # 10
        # 'MPI-ESM-MR',   # 11
        # 'CMCC-CM',      # 12
        'inmcm4',       # 13
        # 'NorESM1-M',    # 14
        # 'CanESM2',      # 15 
        # 'MIROC5',       # 16
        # 'HadGEM2-CC',   # 17
        # 'MRI-CGCM3',    # 18
        'CESM1-BGC'     # 19
        ]
    
    experiments = [
        'historical',
        'rcp85'
        ]
    
    resolutions = [
        # 'orig',
        'regridded'
        ]

    
    for model in models:
        print(model)
        for experiment in experiments:
            print(experiment)

            # ds_pr = get_pr(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_tas = get_tas(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_hus = get_hus(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_hur = get_hur(institutes[model], model, experiment, resolution=resolutions[0])
            ds_wap = get_wap(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_cl, ds_p_hybridsigma = get_cl(institutes[model], model, experiment, resolution=resolutions[0])
    

            save_pr = False
            save_tas = False
            save_hus = False
            save_hur = False
            save_wap = True
            save_cl = False
            
            folder_save = '{}/data/cmip5/ds/'.format(home)
            
            if save_pr:
                fileName = model + '_precip_' + experiment + '_' + resolutions[0] + '.nc'
                save_file(ds_pr, folder_save, fileName)
                
            if save_tas:
                fileName = model + '_tas_' + experiment + '_' + resolutions[0] +  '.nc'
                save_file(ds_tas, folder_save, fileName)

            if save_hus:
                fileName = model + '_hus_' + experiment + '_' + resolutions[0] +  '.nc'
                save_file(ds_hus, folder_save, fileName)

            if save_hur:
                fileName = model + '_hur_' + experiment + '_' + resolutions[0] +  '.nc'
                save_file(ds_hur, folder_save, fileName)
                
            if save_wap:
                fileName = model + '_wap_' + experiment + '_' + resolutions[0] +  '.nc'
                save_file(ds_wap, folder_save, fileName)

            if save_cl:
                fileName = model + '_cl_' + experiment + '_' + resolutions[0] +  '.nc'
                save_file(ds_cl, folder_save, fileName)
                
                fileName = model + '_p_hybridsigma_' + experiment + '_' + resolutions[0] +  '.nc'
                save_file(ds_p_hybridsigma, folder_save, fileName)

















