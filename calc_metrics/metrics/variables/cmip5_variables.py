import xarray as xr
import xesmf as xe
import numpy as np
import os
import scipy



# ------------------------------------ choosing and processing ------------------------------------------------


def concat_files(path_folder, experiment):
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

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    # f0 = path_fileList[0]
    # f1 = path_fileList[1]
    # if (len(path_fileList) == 2) and (int(f0[f0.index(".nc")-8:f0.index(".nc")]) == int(f1[f1.index(".nc")-8:f1.index(".nc")])):
    #     ds = xr.open_dataset(path_fileList[0]).sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))
    # else:
        ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))

    return ds


def regrid_conserv_xesmf(ds_in):
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    ds_out = xr.open_dataset(folder + '/' + fileName).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)

    return regridder


def choose_ensemble(model, variable):
    ensemble = 'r1i1p1'

    if variable == 'pr' or variable == 'tas':
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


def data_exist(model, experiment, variable):
    data_exist = 'yes'

    if variable == 'pw':
        if model == 'CESM1-BGC':
            data_exist = 'no'
        if (model == 'HadGEM2-AO' or model == 'EC-EARTH') and experiment == 'rcp85':
            data_exist = 'no'
    
    if variable == 'hur':
        if model == 'EC-EARTH' and experiment == 'rcp85':
            data_exist = 'no'

    if variable == 'wap':
        if model == 'GISS-E2-H' or model == 'CCSM4' or model == 'HadGEM2-AO' or model == 'inmcm4' or model == 'HadGEM2-CC' or model =='CESM1-BGC' or model == 'EC-EARTH':
            data_exist = 'no'
        if model == 'bcc-csm1-1' and experiment=='rcp85':
            data_exist = 'no'

    if variable == 'cl':
        if model == 'CNRM-CM5' or model == 'CCSM4' or model == 'HadGEM2-AO':
            data_exist = 'no'
        if (model == 'EC-EARTH' or model == 'CESM1-BGC') and experiment == 'rcp85':
            data_exist = 'no'
    
    return data_exist


def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)


# --------------------------------------------- getting variable ----------------------------------------------------------

def get_pr(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    variable = 'pr'
    ensemble = choose_ensemble(model, variable)
    version = pick_latestVersion(path_gen, ensemble)
    path_folder =  os.path.join(path_gen, ensemble, version, variable)

    ds = concat_files(path_folder, experiment) # picks out lat: -35, 35

    regridder = regrid_conserv_xesmf(ds) # define regridder based of grid from other model
    precip = ds['pr']*60*60*24 # convert to mm/day
    precip_n = regridder(precip) # conservatively interpolate to grid from other model, onto lat: -30, 30 (_n is new grid)
    precip_n.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9)

    ds_pr = xr.Dataset(
        data_vars = {'precip': precip_n},
        attrs = ds.attrs
        )
    return ds_pr


def get_tas(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    if model == 'FGOALS-g2':
        path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/mon/atmos/Amon'   
    variable = 'tas'
    ensemble = choose_ensemble(model, variable)
    version = pick_latestVersion(path_gen, ensemble)
    path_folder =  os.path.join(path_gen, ensemble, version, variable)

    ds = concat_files(path_folder, experiment)
    
    regridder = regrid_conserv_xesmf(ds)
    tas = ds['tas']-273.15 # convert to degrees Celsius
    tas_n = regridder(tas)
    tas_n.attrs['units']= '\u00B0C'

    ds_tas = xr.Dataset(
        data_vars = {'tas': tas_n},
        attrs = ds.attrs
        )
    return ds_tas


def get_pw(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    variable = 'hus'
    ensemble = choose_ensemble(model, variable)
    
    if data_exist(model,variable) == 'no':
        ds_pw = xr.Dataset(
            data_vars = {'pw': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        regridder = regrid_conserv_xesmf(ds)
        hus = ds['hus'].sel(plev=slice(850e2,0)) # free troposphere (most values at 1000 hPa over land are NaN)
        hus_n = regridder(hus).fillna(0) # mountains will be NaN for larger values as well, so setting them to zero

        g = 9.8
        pw_n = xr.DataArray(
            data= -scipy.integrate.simpson(hus_n.data, hus_n.plev.data, axis=1, even='last')/g,
            dims=['time','lat', 'lon'],
            coords={'time': hus_n.time.data, 'lat': hus_n.lat.data, 'lon': hus_n.lon.data},
            attrs={'units':'mm',
                   'Description': 'precipitable water from 850-0 hpa'}
            )

        ds_pw = xr.Dataset(
            data_vars = {'pw': pw_n},
            attrs = {'description': 'Precipitable water calculated as the vertically integrated specific humidity (simpson\'s method)'}
            )
    return ds_pw


def get_hur(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    variable = 'hur'
    ensemble = choose_ensemble(model, variable)

    if data_exist(model,variable) == 'no':
        ds_hur = xr.Dataset(
            data_vars = {'hur': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        regridder = regrid_conserv_xesmf(ds)
        hur = ds['hur'].sel(plev=slice(850e2,0)) # already in units of %
        hur_n = regridder(hur) 
        hur_n = (hur_n * ds.plev).sum(dim='plev') / ds.plev.sum(dim='plev')
        hur_n.attrs['units']= '%'
        hur_n.attrs['Description'] = 'weighted mean relative humidity from 850-0 hpa'

        ds_hur = xr.Dataset(
            data_vars = {'hur': hur_n},
            attrs = {'Description': 'weighted mean relative humidity'}
            )
    return ds_hur



def get_wap500(institute, model, experiment):

    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute +'/'+ model +'/'+ experiment +'/day/atmos/day'
    variable = 'wap'
    ensemble = choose_ensemble(model, variable)

    if data_exist(model,variable) == 'no':
        ds_wap500 = xr.Dataset(
            data_vars = {'wap500': np.nan}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        regridder = regrid_conserv_xesmf(ds)
        wap500 = ds['wap'].sel(plev=500e2)*60*60*24/100 # convert to units of hPa/day    
        wap500_n = regridder(wap500)
        wap500_n.attrs['units']= 'hPa day' + chr(0x207B) + chr(0x00B9)

        ds_wap500 = xr.Dataset(
            data_vars = {'wap500': wap500_n}
            )
    return ds_wap500



def get_clouds(institute, model, experiment):
    path_gen = '/g/data/al33/replicas/CMIP5/combined/'+ institute + '/' + model + '/' + experiment + '/mon/atmos/Amon'
    variable = 'cl'
    ensemble = choose_ensemble(model, variable)
        
    if data_exist(model,variable) == 'no':
        ds_clouds = xr.Dataset(
            data_vars = {
                'cloud_low': np.nan, 
                'cloud_high': np.nan},
            attrs = {'description': 'Metric defined as maximum cloud fraction (%) from specified pressure level intervals'}
            )
    else:
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        regridder = regrid_conserv_xesmf(ds)
        clouds = ds['cl'] # already in units of %
        clouds_n = regridder(clouds)

        # different models have different conversions from height coordinate to pressure coordinate. Need to convert from height coordinate matrix to pressure coordinate matrix
        if model == 'IPSL-CM5A-MR' or model == 'MPI-ESM-MR' or model=='CanESM2':
            pressureLevels = ds.ap + ds.b*ds.ps
        elif model == 'FGOALS-g2':
            pressureLevels = ds.ptop + ds.lev*(ds.ps-ds.ptop)
        elif model == 'HadGEM2-CC':
            pressureLevels = ds.lev+ds.b*ds.orog
        else:
            pressureLevels = ds.a*ds.p0 + ds.b*ds.ps
        pressureLevels_n = regridder(pressureLevels)

        pressureLevels_low = xr.where((pressureLevels_n<=10000e2) & (pressureLevels_n>=600), 1, 0) # needs to be hPa
        cloud_low = clouds_n*pressureLevels_low
        cloud_low = cloud_low.max(dim='lev')
        cloud_low.attrs['units'] = '%'
        cloud_low.attrs['description'] = 'Maximum cloud fraction (%) from plev: 1000-600 hpa'

        pressureLevels_high = xr.where((pressureLevels_n<=250e2) & (pressureLevels_n>=100), 1, 0)
        cloud_high = clouds_n*pressureLevels_high
        cloud_high = cloud_high.max(dim='lev')
        cloud_high.attrs['units'] = '%'
        cloud_high.attrs['description'] = 'Maximum cloud fraction (%) from plev: 250-100 hpa'

        ds_clouds = xr.Dataset(
            data_vars = {
                'cloud_low': cloud_low, 
                'cloud_high': cloud_high},
            attrs = {'description': 'Metric defined as maximum cloud fraction (%) from specified pressure level intervals'}
            )
    return ds_clouds



def get_stability(institute, model, experiment):
    print('hello')




if __name__ == '__main__':

    import matplotlib.pyplot as plt

    models = [
            # 'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     # 2
            # 'GISS-E2-H',    # 3
            # 'bcc-csm1-1',   # 4
            # 'CNRM-CM5',     # 5
            # 'CCSM4',        # 6
            # 'HadGEM2-AO',   # 7
            # 'BNU-ESM',      # 8
            # 'EC-EARTH',     # 9
            # 'FGOALS-g2',    # 10
            # 'MPI-ESM-MR',   # 11
            # 'CMCC-CM',      # 12
            # 'inmcm4',       # 13
            # 'NorESM1-M',    # 14
            # 'CanESM2',      # 15 
            # 'MIROC5',       # 16
            # 'HadGEM2-CC',   # 17
            # 'MRI-CGCM3',    # 18
            # 'CESM1-BGC'     # 19
            ]
    
    experiments = [
                'historical',
                # 'rcp85'
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

    for model in models:
        for experiment in experiments:

            ds_pr = get_pr(institutes[model], model, experiment)
            ds_tas = get_tas(institutes[model], model, experiment)
            ds_pw = get_pw(institutes[model], model, experiment)
            ds_hur = get_hur(institutes[model], model, experiment)
            ds_wap500 = get_wap500(institutes[model], model, experiment)
            ds_clouds = get_clouds(institutes[model], model, experiment)
    

            save_pr = False
            save_tas = False
            save_pw = False
            save_hur = False
            save_wap500 = False
            save_cl = False
            
            folder = '/g/data/k10/cb4968/data/cmip5/ds/'
            
            if save_pr:
                fileName = model + '_precip_' + experiment + '.nc'
                save_file(ds_pr, folder, fileName)
                
            if save_tas:
                fileName = model + '_tas_' + experiment + '.nc'
                save_file(ds_tas, folder, fileName)

            if save_pw:
                fileName = model + '_pw_' + experiment + '.nc'
                save_file(ds_pw, folder, fileName)

            if save_hur:
                fileName = model + '_hur_' + experiment + '.nc'
                save_file(ds_hur, folder, fileName)
                
            if save_wap500:
                fileName = model + '_wap500_' + experiment + '.nc'
                save_file(ds_wap500, folder, fileName)

            if save_cl:
                fileName = model + '_clMax_' + experiment + '.nc'
                save_file(ds_clouds, folder, fileName)

















