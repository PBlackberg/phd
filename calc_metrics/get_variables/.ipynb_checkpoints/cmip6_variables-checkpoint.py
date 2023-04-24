import xarray as xr
import xesmf as xe
import numpy as np
import os
import scipy



# --------------------------------------------------------- finding ensemble/version and concatenating files --------------------------------------------------------

def data_exist(dataset, experiment, variable):
    data_exist = 'yes'

    return data_exist


def choose_ensemble(model, experiment, variable):
    ensemble = 'r1i1p1f1'
    
    if model == 'CNRM-CM6-1' or model =='UKESM1-0-LL':
        ensemble = 'r1i1p1f2'

    return ensemble


def last_letters(model, experiment, variable):
    folder = 'gn'

    if model == 'CNRM-CM6-1':
        folder = 'gr'

    if model == 'GFDL-CM4':
        folder = 'gr2'
        
    return folder


def pick_latestVersion(path_gen):
    versions = os.listdir(path_gen)
    if len(versions)>1:
        version = max(versions, key=lambda x: int(x[1:]))
    else:
        version = versions[0]
    return version


def rcp_years(model):
    yearEnd_first = 1970
    yearStart_last = 1999
    
    if model == 'FGOALS-g3':
        yearEnd_first = 470
        yearStart_last = 499
    
    if model == 'NorESM2-MM' or model == 'GFDL-CM4' or  model =='CESM2':
        yearEnd_first = 470
        yearStart_last = 499
        
    if model == 'MIROC6':
        yearEnd_first = 3270
        yearStart_last = 3399
        
    return str(yearEnd_first), str(yearStart_last)
    
    
def concat_files(path_folder, experiment):
    if experiment == 'historical':
        yearEnd_first = 1970
        yearStart_last = 1999

    if experiment == 'abrupt-4xCO2':
        yearEnd_first, yearStart_last = rcp_years(model)

    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if 'Amon' in path_folder:
        files = sorted(files, key=lambda x: x[x.index(".nc")-13:x.index(".nc")-9])
        files = [f for f in files if int(f[f.index(".nc")-13:f.index(".nc")-9]) <= int(yearStart_last) and int(f[f.index(".nc")-6:f.index(".nc")-2]) >= int(yearEnd_first)]
    else:
        files = sorted(files, key=lambda x: x[x.index(".nc")-17:x.index(".nc")-13])
        files = [f for f in files if int(f[f.index(".nc")-17:f.index(".nc")-13]) <= int(yearStart_last) and int(f[f.index(".nc")-8:f.index(".nc")-4]) >= int(yearEnd_first)]

        for f in files:
            if int(f[f.index(".nc")-17:f.index(".nc")-9])==19790101 and int(f[f.index(".nc")-8:f.index(".nc")])==20051231:
                files.remove(f)

    path_fileList = []
    for file in files:
        path_fileList = np.append(path_fileList, os.path.join(path_folder, file))

    ds = xr.open_mfdataset(path_fileList, combine='by_coords').sel(time=slice(str(yearEnd_first), str(yearStart_last)),lat=slice(-35,35))
    return ds


def regrid_conserv_xesmf(ds_in):
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr' # interpolate grid to grid from FGOALS-g2
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'

    ds_out = xr.open_dataset('{}/{}'.format(folder, fileName)).sel(lat=slice(-30,30))
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)
    return regridder


def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    dataset.to_netcdf(path)


# -------------------------------------------------------------------------------- get variable -------------------------------------------------------------------------------

def get_pr(institute, model, experiment, resolution):
    variable = 'pr'
    if data_exist(model, experiment, variable) == 'no':
        ds_pr = xr.Dataset(
            data_vars = {'precip': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment, variable)
        path_gen = '/g/data/oi10/replicas/CMIP6/CMIP/{}/{}/{}/{}/day/pr'.format(institute, model, experiment, ensemble)
        
        cmip6_feature = last_letters(model, experiment, variable)
        version = pick_latestVersion(os.path.join(path_gen, cmip6_feature))
        path_folder =  os.path.join(path_gen, cmip6_feature, version)

        ds = concat_files(path_folder, experiment) # picks out lat: -35, 35

        precip = ds['pr']*60*60*24 # convert to mm/day
        precip.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9) # give new units
        
        if resolution == 'original':
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
    if data_exist(model, experiment, variable) == 'no':
        ds_tas = xr.Dataset(
            data_vars = {'precip': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment, variable)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        tas = ds['tas']-273.15 # convert to degrees Celsius
        tas.attrs['units']= '\u00B0C'

        if resolution == 'original':
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
    
    variable = 'hus'
    if data_exist(model,variable) == 'no':
        ds_hus = xr.Dataset(
            data_vars = {'hus': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment, variable)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        hus = ds['hus'] # unitless kg/kg

        if resolution == 'original':
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
    if data_exist(model,variable) == 'no':
        ds_hur = xr.Dataset(
            data_vars = {'hur': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, experiment, variable)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        hur = ds['hur'] # units in %
        hur.attrs['units']= '%'

        if resolution == 'original':
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

    variable = 'wap'
    if data_exist(model,variable) == 'no':
        ds_wap = xr.Dataset(
            data_vars = {'wap': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, variable)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        wap = ds['wap']*60*60*24/100 # convert to hPa/day   
        wap.attrs['units']= 'hPa day' + chr(0x207B) + chr(0x00B9) 

        if resolution == 'original':
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
    if data_exist(model,variable) == 'no':
        ds_cl = xr.Dataset(
            data_vars = {'cl': np.nan}
            )
        ds_p_hybridsigma = xr.Dataset(
            data_vars = {'p_hybridsigma': np.nan}
            )
    else:
        ensemble = choose_ensemble(model, variable)
        version = pick_latestVersion(path_gen, ensemble)
        path_folder =  os.path.join(path_gen, ensemble, version, variable)

        ds = concat_files(path_folder, experiment)
        
        if resolution == 'original':
            ds_cl = ds # units in % on sigma pressure coordinates

            if model == 'IPSL-CM5A-MR' or model == 'MPI-ESM-MR' or model=='CanESM2': # different models have different conversions from height coordinate to pressure coordinate.
                p_hybridsigma = ds.ap + ds.b*ds.ps
            elif model == 'FGOALS-g2':
                p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
            elif model == 'HadGEM2-CC':
                p_hybridsigma = ds.lev+ds.b*ds.orog
            else:
                p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps
            
            ds_p_hybridsigma = xr.Dataset(
                data_vars = {
                    'p_hybridsigma': p_hybridsigma},
                attrs = ds.attrs
                )

        elif resolution == 'regridded':
            cl = ds['cl'] # units in % on sigma pressure coordinates

            regridder = regrid_conserv_xesmf(ds)
            cl_n = regridder(cl)
            p_hybridsigma_n = regridder(p_hybridsigma)

            ds_cl = xr.Dataset(
                data_vars = {
                    'cl': cl_n},
                attrs = ds.attrs
                )
            ds_p_hybridsigma = xr.Dataset(
                data_vars = {
                    'p_hybridsigma': p_hybridsigma_n},
                attrs = ds.attrs
                )
    return ds_cl, ds_p_hybridsigma






if __name__ == '__main__':
    import matplotlib.pyplot as plt

    
    models = [
        # 'TaiESM1',        # 1 # rcp monthly
        # 'BCC-CSM2-MR',    # 2 # rcp monthly   
        'FGOALS-g3',        # 3 # rcp 0463 - 0614
        'CNRM-CM6-1',     # 4 # rcp 1850-1999
        'MIROC6',         # 5 # rcp 3200 - 3340
        'MPI-ESM1-2-HR',  # 6 # rcp 1850 - 2014
        'NorESM2-MM',     # 7 # rcp 0001 - 0141
        'GFDL-CM4',       # 8 # rcp 0001 - 0141 (gr2)
        'CanESM5',        # 9 # rcp 1850 - 2000
        # 'CMCC-ESM2',      # 10 # rcp monthly
        'UKESM1-0-LL',    # 11 # rcp 1850 - 1999
        'MRI-ESM2-0',     # 12 # rcp 1850 - 2000
        'CESM2',          # 13 # rcp 0001 - 0990  (multiple fill values (check if all get converted to NaN), for historical)
        'NESM3',          # 12 # rcp 1850-2014
            ]


    resolutions = [
        'original',
        # 'regridded'
        ]

    experiments = [
                # 'historical',
                'abrupt-4xCO2'
                ]

    institutes = {
        'TaiESM1':'AS-RCEC',
        'BCC-CSM2-MR':'BCC',
        'FGOALS-g3':'CAS',
        'CNRM-CM6-1':'CNRM-CERFACS',
        'MIROC6':'MIROC',
        'MPI-ESM1-2-HR':'MPI-M',
        'GISS-E2-1-H':'NASA-GISS',
        'NorESM2-MM':'NCC',
        'GFDL-CM4':'NOAA-GFDL',
        'CanESM5':'CCCma',
        'CMCC-ESM2':'CMCC',
        'UKESM1-0-LL':'MOHC',
        'MRI-ESM2-0':'MRI',
        'CESM2':'NCAR',
        'NESM3':'NUIST'
        }


    # not included:
    # 'IITM-ESM':'CCCR-IITM'
    # 'EC-Earth3':'EC-Earth-Consortium'
    # 'HAMMOZ-Consortium':'MPI-ESM-1-2-HAM'
    # 'IPSL-CM6A-LR':'IPSL'
    # 'GISS-E2-1-H':'NASA-GISS' (only monthly for all variables)
    # 'SNU':'SAM0-UNICON'
    # 'MCM-UA-1-0':UA
    # 'AWI-CM-1-1-MR':AWI
    # 'CAMS-CSM1-0':'CAMS'
    # 'E3SM-1-0':'E3SM-Project'
    # 'FIO-ESM-2-0':'FIO-QLNM'
    # 'INM-CM5-0':'INM'
    # 'KIOST-ESM':'KIOST'
    # 'KACE-1-0-G':'NIMS-KMA' (this institute has data for UKESM1-0-LL which is already included from a different institute)
    # 'CIESM':'THU'
    


    for model in models:
        print(model)
        for experiment in experiments:

            ds_pr = get_pr(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_tas = get_tas(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_hus = get_hus(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_hur = get_hur(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_wap = get_wap(institutes[model], model, experiment, resolution=resolutions[0])
            # ds_cl, ds_p_hybridsigma = get_cl(institutes[model], model, experiment, resolution=resolutions[0])
    

            save_pr = True
            save_tas = False
            save_hus = False
            save_hur = False
            save_wap = False
            save_cl = False
            
            folder = '/g/data/k10/cb4968/data/cmip6/ds/precip'
            
            if save_pr:
                fileName = model + '_precip_' + experiment + '.nc'
                save_file(ds_pr, folder, fileName)
                
            if save_tas:
                fileName = model + '_tas_' + experiment + '.nc'
                save_file(ds_tas, folder, fileName)

            if save_hus:
                fileName = model + '_hus_' + experiment + '.nc'
                save_file(ds_hus, folder, fileName)

            if save_hur:
                fileName = model + '_hur_' + experiment + '.nc'
                save_file(ds_hur, folder, fileName)
                
            if save_wap:
                fileName = model + '_wap_' + experiment + '.nc'
                save_file(ds_wap, folder, fileName)

            if save_cl:
                fileName = model + '_cl_' + experiment + '.nc'
                save_file(ds_cl, folder, fileName)
                
                fileName = model + '_p_hybridsigma_' + experiment + '.nc'
                save_file(ds_p_hybridsigma, folder, fileName)











































































