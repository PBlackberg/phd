import xarray as xr
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/switch')
import myFuncs as mF
import myVars as mV
sys.path.insert(0, f'{os.getcwd()}/util')
import get_data as gD



# -----------------------------------
#  Vertically interp. cloud fraction
# -----------------------------------
# ---------------------------------------------------------------------------------------- With scipy.interp1d() ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def interp_z_to_p(switch, da, z, z_new): 
    ''' From hybrid-sigma pressure levels to height coordinates [m], later converted to pressure [hPa] 
    da    - xarray data array (dim = (time, lev, lat, lon))
    z     - xarray data array (dim = (lev, lat, lon))
    z_new - numpy array       (dim = (lev))
    '''
    print(f'Column cloud fraction value before interp:\n {da.isel(time=0, lat = int(len(da.lat)/2), lon = 5).values} \n at heights [m]: \n {z.isel(lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}') if switch['show_one_column'] else None
    interpolated_data = np.empty((len(da['time']), len(z_new), len(da['lat']), len(da['lon'])))
    for time_i in range(len(da['time'])):
        print(f'month:{time_i} started')
        for lat_i in range(len(da['lat'])):
            for lon_i in range(len(da['lon'])):
                z_1d = z.sel(lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                var_1d = da.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                f = interp1d(z_1d, var_1d, kind='linear', bounds_error=False, fill_value=0)     
                var_1d_interp = f(z_new)                                                                          
                interpolated_data[time_i, :, lat_i, lon_i] = var_1d_interp
    da_z_fixed = xr.DataArray(interpolated_data, dims=('time', 'lev', 'lat', 'lon'), coords={'time': da['time'], 'lev': z_new, 'lat': da['lat'], 'lon': da['lon']})
    da_z_fixed['lev'] = 101325*(1-((2.25577e-5)*da_z_fixed['lev']))**(5.25588)
    da_z_fixed = da_z_fixed.rename({'lev':'plev'})
    print(f'Column cloud fraction value after interp to pressure coordinates:\n {da_z_fixed.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp heights [hPa]: \n {da_z_fixed.plev.values}') if switch['show_one_column'] else None
    return da_z_fixed

@mF.timing_decorator
def interp_p_to_p_new(switch, da, p, p_new): 
    ''' From pressure levels to common pressure levels (when pressure has same dimensions as variable) 
    da    - xarray data array (dim = (time, plev, lat, lon))
    p     - xarray data array (dim = (lev, time, lat, lon))
    p_new - numpy array       (dim = (plev))
    '''
    print(f'Column cloud fraction value before interp to new pressure coordinates:\n {da.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at pressure [hPa]: \n {p.isel(lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}') if switch['show_one_column'] else None
    interpolated_data = np.empty((len(da['time']), len(p_new), len(da['lat']), len(da['lon'])))
    for time_i in range(len(da['time'])):
        print(f'month:{time_i} started')
        for lat_i in range(len(da['lat'])):
            for lon_i in range(len(da['lon'])):
                p_1d = p.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                var_1d = da.sel(time=da['time'][time_i], lon=da['lon'][lon_i], lat=da['lat'][lat_i]).values
                f = interp1d(p_1d, var_1d, kind='linear', bounds_error=False, fill_value=0)     
                var_1d_interp = f(p_new)                                                                          
                interpolated_data[time_i, :, lat_i, lon_i] = var_1d_interp                                       
    da_p_new = xr.DataArray(interpolated_data, dims=('time', 'plev', 'lat', 'lon'), coords={'time': da['time'], 'plev': p_new, 'lat': da['lat'], 'lon': da['lon']})
    print(f'Column cloud fraction value after interp to new pressure coordinates:\n {da_p_new.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp pressure [hPa]: \n {da_p_new.plev.values}') if switch['show_one_column'] else None
    return da_p_new


# ------------------------------------------------------------------------------------------ With xarray in-built ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def interp_p_to_p_new_xr(switch, da, p_new): # does the same thing as with numpy method, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (when pressure is 1D) 
    da    - xarray data array (dim = (time, plev, lat, lon))
    p_new - numpy array       (dim = (plev))
    '''
    print(f'Column cloud fraction value before interp to new pressure coordinates:\n {da.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp pressure [hPa]: \n {da.plev.values}') if switch['show_one_column'] else None
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})    
    warnings.resetwarnings()
    print(f'Column cloud fraction value after interp to new pressure coordinates:\n {da_p_new.isel(time=0, lat = int(len(da.lat)/2), lon = int(len(da.lon)/2)).values}, \n at interp pressure [hPa]: \n {da_p_new.plev.values}') if switch['show_one_column'] else None
    return da_p_new



# ----------------------------------
#           load data
# ----------------------------------
def load_data(switch, dataset, source, experiment):
    ds = gD.get_cmip_data('ds_cl', source, dataset, experiment).isel(time=slice(0,2))                                                                                 if switch['gadi_data']   else None
    ds = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/cl/{source}/{dataset}_ds_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc').isel(time=slice(0,2)) if switch['sample_data'] else da 
    da = ds['cl']
    return ds, da



# ----------------------------------
#    run dataset / experiment
# ----------------------------------
def save_interp_data(switch, dataset, source, experiment, ds):
    folder = f'{mV.folder_save[0]}/sample_data/cl/{source}'
    filename = f'{dataset}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename) if switch['save'] else None

def run_interp(switch, dataset, source, experiment, ds, da, z_new, p_new):
    if dataset == 'IITM-ESM':               
        pass                                                    # already on pressure levels (19 levels)

    elif dataset == 'IPSL-CM6A-LR':         
        da = da.rename({'presnivs':'plev'})
        da = interp_p_to_p_new_xr(switch, da, p_new)            # already on pressure levels (79 levels)

    elif dataset in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'IPSL-CM5A-MR', 'MPI-ESM-MR', 'CanESM2']: 
        p = ds['ap'] + ds['b']*ds['ps']                         # same shape as variable
        da = interp_p_to_p_new(switch, da, p, p_new)

    elif dataset in ['FGOALS-g2', 'FGOALS-g3']:                                                   
        p = ds['ptop'] + ds['lev']*(ds['ps']-ds['ptop'])        # same shape as variable
        da = da.rename({'lev':'plev'})
        da = interp_p_to_p_new(switch, da, p, p_new)

    elif dataset in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'HadGEM2-CC']:
        z = ds['lev']+ds['b']*ds['orog']
        da = interp_z_to_p(switch, da, z, z_new)
        da = interp_p_to_p_new_xr(switch, da, p_new)

    else:
        p = ds['a']*ds['p0'] + ds['b']*ds['ps']                 # same shape as variable
        da = da.rename({'lev':'plev'})
        da = interp_p_to_p_new(switch, da, p, p_new)

    save_interp_data(switch, dataset, source, experiment, ds = xr.Dataset(data_vars = {'cl': da}))

def run_experiment(switch, dataset, source):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment):
            continue
        print(f'\t\t {experiment}')
        ds, da = load_data(switch, dataset, source, experiment)
        z_new = np.linspace(0, 15000, 30)
        p_new = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])
        run_interp(switch, dataset, source, experiment, ds, da, z_new, p_new)

def run_dataset(switch):
    print(f'Vertically regridding {mV.resolutions[0]} {mV.timescales[0]} cloudfraction data')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'\t{dataset} ({source})')
        run_experiment(switch, dataset, source)



# ----------------------------------
#  Choose settings and run script
# ----------------------------------
if __name__ == '__main__':

    switch = {
            'sample_data':     True, 'gadi_data': False,                                            # data to use
            'show_one_column': True,                                                               # example of what interpolation does to cloud fraction in one column
            'save':            False                                                                # save interpolated data
            }
    run_dataset(switch)





















