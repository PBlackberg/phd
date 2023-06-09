import xarray as xr
import numpy as np
import concat_files as cfiles


# make one function for cmip6, one for cmip5, and one or more for obs, then one function that calls them depending on switch


def prData_exist(dataset, experiment):
    ''' Check if model/project has data'''
    data_exist = 'True'
    return data_exist


def get_pr(institute, model, experiment, resolution):
    ''' Surfae precipitation (daily) '''
    variable = 'pr'
    if not prData_exist(model, experiment, variable):
        ds_pr = xr.Dataset({'precip': np.nan})
    else:
        ensemble = cfiles.choose_ensemble(model, experiment)
        if experiment == 'historical':
            path_gen = '/g/data/oi10/replicas/CMIP6/CMIP/{}/{}/{}/{}/day/{}'.format(institute, model, experiment, ensemble, variable)
        elif experiment == 'ssp585':
            if model == 'MPI-ESM1-2-HR':
                model ='MPI-ESM1-2-LR'
            path_gen = '/g/data/oi10/replicas/CMIP6/ScenarioMIP/{}/{}/{}/{}/day/{}'.format(institute, model, experiment, ensemble, variable)

                    
        ds = cfiles.concat_files(path_gen, experiment, model, variable) # picks out lat: -35, 35
        precip = ds['pr']*60*60*24 # convert to mm/day
        precip.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9) # give new units
        
        if resolution == 'original':
            ds_pr = xr.Dataset(
                data_vars = {'precip': precip},
                attrs = ds.attrs
                )
        elif resolution == 'regridded':
            import xesmf_regrid as regrid
            regridder = regrid.regrid_conserv_xesmf(ds) # define regridder based of grid from other model
            precip_n = regridder(precip) # conservatively interpolate to grid from other model, onto lat: -30, 30 (_n is new grid)
            precip_n.attrs['units']= 'mm day' + chr(0x207B) + chr(0x00B9) # give new units
            
            ds_pr = xr.Dataset(
                data_vars = {'precip': precip_n},
                attrs = ds.attrs
                )
    return ds_pr





























