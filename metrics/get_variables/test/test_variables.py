import xarray as xr
import xesmf as xe
import numpy as np
import os
import scipy
import timeit
home = '/g/data/k10/cb4968'


def data_exist(model, experiment, variable):
    data_exist = 'yes'

    if variable == 'hus':
        if model == 'CESM1-BGC' or model == 'EC-EARTH':
            data_exist = 'no'
        if model == 'HadGEM2-AO' and experiment == 'rcp85':
            data_exist = 'no'
    
    if variable == 'hur':
        if model == 'EC-EARTH':
            data_exist = 'no'

    if variable == 'wap':
        if model == 'GISS-E2-H' or model == 'CCSM4' or model == 'HadGEM2-AO' or model == 'inmcm4' or model == 'HadGEM2-CC' or model =='CESM1-BGC' or model == 'EC-EARTH':
            data_exist = 'no'
        if model == 'bcc-csm1-1' and experiment=='rcp85':
            data_exist = 'no'

    if variable == 'cl' or variable == 'cloud_low' or variable =='cloud_high':
        if model == 'GISS-E2-H' or model == 'CNRM-CM5' or model == 'CCSM4' or model == 'EC-EARTH' or model == 'HadGEM2-AO':
            data_exist = 'no'
        if model == 'CESM1-BGC' and experiment == 'rcp85':
            data_exist = 'no'
    
    return data_exist



def choose_ensemble(model, experiment, variable):
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
    versions = os.listdir(os.path.join(path_gen, ensemble))
    if len(versions)>1:
        version = max(versions, key=lambda x: int(x[1:]))
    else:
        version = versions[0]
    return version


def concat_files(path_folder, model, experiment, variable):
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
        files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
        files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
        files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= yearStart_last and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= yearEnd_first]

        for f in files:
            if int(f[f.index(".nc")-17:f.index(".nc")-9])==19790101 and int(f[f.index(".nc")-8:f.index(".nc")])==20051231:
                files.remove(f)
        
        if model == 'EC-EARTH' and experiment=='historical' and variable=='tas':
            files = files[1:-5]

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

        ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))

    return ds


def regrid_conserv_xesmf(ds_in):
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    ds_out = xr.open_dataset(folder + '/' + fileName).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)

    return regridder


def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)



models = [
    'IPSL-CM5A-MR', # 1
    'GFDL-CM3',     # 2
    'GISS-E2-H',    # 3
    'bcc-csm1-1',   # 4
    'CNRM-CM5',     # 5
    'CCSM4',        # 6
    'HadGEM2-AO',   # 7
    'BNU-ESM',      # 8
    'EC-EARTH',     # 9
    'FGOALS-g2',    # 10
    'MPI-ESM-MR',   # 11
    'CMCC-CM',      # 12
    'inmcm4',       # 13
    'NorESM1-M',    # 14
    'CanESM2',      # 15 
    'MIROC5',       # 16
    'HadGEM2-CC',   # 17
    'MRI-CGCM3',    # 18
    'CESM1-BGC'     # 19
    ]    


institutes = {
    'IPSL-CM5A-MR':'IPSL',
    'GFDL-CM3':'NOAA-GFDL',
    'GISS-E2-H':'NASA-GISS',
    'bcc-csm1-1':'BCC',
    'CNRM-CM5':'CNRM-CERFACS',
    'CCSM4':'NCAR',
    'HadGEM2-AO':'NIMR-KMA',
    'BNU-ESM':'BNU',
    'EC-EARTH':'ICHEC',
    'FGOALS-g2':'LASG-CESS',
    'MPI-ESM-MR':'MPI-M',
    'CMCC-CM':'CMCC',
    'inmcm4':'INM',
    'NorESM1-M':'NCC',
    'CanESM2':'CCCma',
    'MIROC5':'MIROC',
    'HadGEM2-CC':'MOHC',
    'MRI-CGCM3':'MRI',
    'CESM1-BGC':'NSF-DOE-NCAR'
}
    
    
    
experiments = [
        'historical',
        'rcp85'
            ]

    
def get_cl(institute, model, experiment):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    variable = 'cl'
    ensemble = choose_ensemble(model, experiment, variable)
        
    if data_exist(model,experiment,variable) == 'no':
        ds_cl = xr.Dataset(
            data_vars = {
                'cl': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds_cl = concat_files(path_folder, model, experiment, variable)
        
    return ds_cl


def get_wap(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    variable = 'wap'
    ensemble = choose_ensemble(model, experiment, variable)

    if data_exist(model,experiment,variable) == 'no':
        ds_wap = xr.Dataset(
            data_vars = {'wap': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, model, experiment, variable)
        
        regridder = regrid_conserv_xesmf(ds)
        wap = ds['wap'] #.sel(plev=500e2)*60*60*24/100 # convert to units of hPa/day    
        wap = regridder(wap)
        wap.attrs['units']= 'hPa day' + chr(0x207B) + chr(0x00B9)

        ds_wap = xr.Dataset(
            data_vars = {'wap': wap}
            )
    return ds_wap


def get_hur(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    variable = 'hur'
    ensemble = choose_ensemble(model, experiment, variable)

    if data_exist(model,experiment,variable) == 'no':
        ds_hur = xr.Dataset(
            data_vars = {'hur': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, model, experiment, variable)
        
        regridder = regrid_conserv_xesmf(ds)
        hur = ds['hur']
        hur = regridder(hur) 
        hur.attrs['units']= '%'
        hur.attrs['Description'] = 'relative humidity'

        ds_hur = xr.Dataset(
            data_vars = {'hur': hur},
            attrs = {'Description': 'relative humidity'}
            )
    return ds_hur



def get_hus(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    variable = 'hus'
    ensemble = choose_ensemble(model, experiment, variable)
    
    if data_exist(model,experiment,variable) == 'no':
        ds_hus = xr.Dataset(
            data_vars = {'hus': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, model, experiment, variable)
        
        regridder = regrid_conserv_xesmf(ds)
        hus = ds['hus'] #.sel(plev=slice(850e2,0)) # free troposphere (most values at 1000 hPa over land are NaN)
        hus = regridder(hus) #.fillna(0) # mountains will be NaN for larger values as well, so setting them to zero

        ds_hus = xr.Dataset(
            data_vars = {'hus': hus},
            # attrs = {'description': 'Precipitable water calculated as the vertically integrated specific humidity (simpson\'s method)'}
            )
    return ds_hus



start = timeit.default_timer()
for model in models:
    
    print('{}: started'.format(model)) 
    for experiment in experiments:
        
        # ds_cl = get_cl(institutes[model], model, experiment)
        # ds_wap = get_wap(institutes[model], model, experiment)
        # ds_hur = get_hur(institutes[model], model, experiment)
        ds_hus = get_hus(institutes[model], model, experiment)

        folder_save = '/g/data/k10/cb4968/data/cmip5/ds_cmip5/clouds'
        save = True

        if save:
            # fileName = model + '_cl_' + experiment + '.nc'
            # save_file(ds_cl, folder_save, fileName)


            # fileName = model + '_wap_' + experiment + '.nc'
            # save_file(ds_wap, folder_save, fileName)

            # fileName = model + '_hur_' + experiment + '.nc'
            # save_file(ds_hur, folder_save, fileName)

                fileName = model + '_hus_' + experiment + '.nc'
                save_file(ds_hus, folder_save, fileName)
    
    print('{}: finished'.format(model))

stop = timeit.default_timer()
print('{}: finsihed (took {} minutes)'.format(model, (stop-start)/60))






































































