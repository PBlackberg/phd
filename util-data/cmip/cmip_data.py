''' 
# ------------------------
#       Get data
# ------------------------
This script loads cmip data from nci and remaps to common grid.
The data is stored temporarily in scratch directory.
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import warnings
from pathlib import Path


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD                          # chosen datasets
sys.path.insert(0, f'{os.getcwd()}/util-data')
import cmip.xesmf_regrid as rGh                       # regridding
# import cmip.cl_cmip_regridVert as rGh               # regridding hybrid sigma coordinates
import cmip.model_institutes as mI
import missing_data as mD
sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF


# ------------------------
#      Find folders
# ------------------------
# ---------------------------------------------------------------------------------------- folders ----------------------------------------------------------------------------------------------------------#
def find_source(dataset):
    '''Determining source of dataset '''
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else None      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source     
    return source

def folder_ensemble(source, model, experiment):
    ''' Some models don't have the ensemble most common amongst other models and some models don't have common ensembles for historical and warmer simulation '''
    if source == 'cmip5':
        ensemble = 'r6i1p1'    if model in ['EC-EARTH', 'CCSM4']                                                        else 'r1i1p1'
        ensemble = 'r6i1p1'    if model in ['GISS-E2-H'] and experiment == 'historical'                                 else ensemble
        ensemble = 'r2i1p1'    if model in ['GISS-E2-H'] and not experiment == 'historical'                             else ensemble
    if source == 'cmip6':
        ensemble = 'r1i1p1f2'  if model in ['CNRM-CM6-1', 'UKESM1-0-LL', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'MIROC-ES2L']  else 'r1i1p1f1'
        ensemble = 'r11i1p1f1' if model == 'CESM2' and not experiment == 'historical'                                   else ensemble
    return ensemble

def folder_grid(model):
    ''' Some models have a different grid folder in the path to the files (only applicable for cmip6) '''
    folder = 'gn'
    folder = 'gr'  if model in ['CNRM-CM6-1', 'EC-Earth3', 'IPSL-CM6A-LR', 'FGOALS-f3-L', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'KACE-1-0-G'] else folder
    folder = 'gr1' if model in ['GFDL-CM4', 'INM-CM5-0', 'KIOST-ESM', 'GFDL-ESM4', 'INM-CM4-8'] else folder           
    return folder

def folder_latestVersion(path):
    ''' Picks the latest version if there are multiple '''    
    versions = os.listdir(path)
    version = max(versions, key=lambda x: int(x[1:])) if len(versions)>1 else versions[0]
    return version

def folder_var(var, model, experiment, ensemble, project, timeInterval):
    ''' ACCESS model is stored separately '''
    if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
        path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mI.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}'
        folder_to_grid = folder_grid(model)
        version = 'latest'
    else:
        path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mI.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{var}' 
        folder_to_grid = folder_grid(model)
        version = folder_latestVersion(os.path.join(path_gen, folder_to_grid))
    return f'{path_gen}/{folder_to_grid}/{version}'



# ------------------------
#   Remap coordinates
# ------------------------
# --------------------------------------------------------------------------------- remap ----------------------------------------------------------------------------------------------------------#
def regrid_hor(ds, da): 
    ''' Regrid to common horizontal grid (ds and da from the same model)'''
    regridder = rGh.regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    da = regridder(da)
    return da

def regrid_vert(da, model = ''):                                                                                # does the same thing as scipy.interp1d, but quicker (can only be applied for models with 1D pressure coordinate)
    ''' Interpolate to common pressure levels (cloud fraction is dealt with separately)'''
    da['plev'] = da['plev'].round(0)                if model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else da['plev'] # plev coordinate is specified to a higher number of significant figures in these models
    p_new = np.array([100000, 92500, 85000, 70000, 60000, 50000, 40000, 30000, 25000, 20000, 15000, 10000, 7000, 5000, 3000, 2000, 1000, 500, 100])      

    warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
    da_p_new = da.interp(plev=p_new, method='linear', kwargs={'bounds_error':False, "fill_value": 0})    
    warnings.resetwarnings()
    return da_p_new
                                                                
def get_p_hybrid(model, ds):
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''
    if model == 'IITM-ESM':               
        da = ds['plev']
    elif model == 'IPSL-CM6A-LR':         
        da = ds['presnivs']
    elif model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR', 'IPSL-CM5A-MR', 'MPI-ESM-MR', 'CanESM2']: 
        da = ds['ap'] + ds['b']*ds['ps']
    elif model in ['FGOALS-g2', 'FGOALS-g3']:                                                   
        da = ds['ptop'] + ds['lev']*(ds['ps']-ds['ptop'])
    elif model in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5', 'HadGEM2-CC']:
        da = ds['lev']+ds['b']*ds['orog']
    else:
        da = ds['a']*ds['p0'] + ds['b']*ds['ps']
    return da



# ------------------------
#        Get data
# ------------------------
# ----------------------------------------------------------------------------------- open files ----------------------------------------------------------------------------------------------------------#
def concat_data(path_folder, experiment):
    ''' Concatenate files between specified years '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    year1, year2 = (1970, 1999)                      if experiment == 'historical' else (2070, 2099)                # range of years to concatenate files for
    fileYear1_charStart, fileYear1_charEnd = (13, 9) if 'Amon' in path_folder      else (17, 13)                    # character index range for starting year of file (counting from the end)
    fileYear2_charStart, fileYear2_charEnd = (6, 2)  if 'Amon' in path_folder      else (8, 4)                      #                             end
    files = sorted(files, key=lambda x: x[x.index(".nc")-fileYear1_charStart:x.index(".nc")-fileYear1_charEnd])
    files = [f for f in files if int(f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]) <= int(year2) and int(f[f.index(".nc")-fileYear2_charStart : f.index(".nc")-fileYear2_charEnd]) >= int(year1)]
    paths = []
    for file in files:
        paths = np.append(paths, os.path.join(path_folder, file))
    # print(paths[0])                                                                                                
    ds = xr.open_mfdataset(paths, combine='by_coords').sel(time=slice(str(year1), str(year2)),lat=slice(-35,35))     # take out a little bit wider range to not exclude data when interpolating grid
    return ds


# ----------------------------------------------------------------------------- get cloud fraction variable ----------------------------------------------------------------------------------------------------------#
def get_cmip_cl_data(var_name, model, experiment, resolution, timescale, source, ensemble, project, time_interval): 
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''       
    if var_name in ['cl']:                                  # interpolation takes a long time, so stored locally       
        path = f'/g/data/k10/cb4968/data/sample_data/cl/{source}/{model}_cl_{timescale}_{experiment}_{resolution}.nc'                                                                                            
        return xr.open_dataset(path) #['cl']         
    else:                                                   # for interpolation in 'cl_vert_interp' in util folder (could extend to other variables on hybrid-sigma pressure coordinates)
        print('need to fx function first')
        ds = ''
        # path_folder = folder_var('cl', model, experiment, ensemble, project, time_interval)
        # ds = concat_data(path_folder, experiment) 
        # if var_name == 'ds_cl':
        #         return ds                
        # if var_name == 'cl_p_hybrid':
        #     da = ds['cl']     
        # if var_name == 'p_hybrid':
        #     da = get_p_hybrid(model, ds)
        #     if model in ['IITM-ESM', 'IPSL-CM6A-LR']:
        #         return xr.Dataset(data_vars = {f'{var_name}': da}, attrs = ds.attrs)                # no lat, lon for vertical coordiante for these models
        # da = regrid_hor(ds, da)     if cD.resolutions[0] == 'regridded' else da                     # horizontally interpolate
        # ds = xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)   # if regridded it should already be lat: [-30,30]
    return ds


# --------------------------------------------------------------------------------- get variables ----------------------------------------------------------------------------------------------------------#
def get_var_data(var_name, model, experiment, resolution, timescale, source):
    ''' concatenate files, interpolate grid, and mask '''
    ensemble = folder_ensemble(source, model, experiment)
    project = 'CMIP'        if experiment == 'historical'  else 'ScenarioMIP'
    time_interval = 'day'    if timescale == 'daily' else 'Amon'         
    if var_name in ['cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid']:                                                  # cloud variable is on hybrid-sigma pressure coordiantes, so treated separately
        return get_cmip_cl_data(var_name, model, experiment, resolution, timescale, source, ensemble, project, time_interval)
    else:                                                                                                       # all other 2D or 3D pressure level variables     
        folder_to_var = folder_var(var_name, model, experiment, ensemble, project, time_interval)
        # path_folder = '/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical/r11i1p1f1/Amon/clwvi/gr/v20200201/'
        ds = concat_data(folder_to_var, experiment)    
        da = ds[var_name]      
        da = regrid_vert(da, model)                         if 'plev' in da.dims                 else da  # vertically interpolate (some models have different number of vertical levels)
        da = regrid_hor(ds, da)                             if resolution == 'regridded'         else da # horizontally interpolate
    return xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)                 # for when running original resolution. If regridded it should already be lat: [-30,30]



# ----------------------------
#  Run script / save variable
# ----------------------------
# ---------------------------------------------------------------------------------- save ----------------------------------------------------------------------------------------------------- #
def convert_units(model, da, var_name):
    if var_name == 'pr':
        da = da * 60 * 60 * 24
    if var_name == 'tas':
        da = da - 273.15  
    if var_name == 'wap':
        da = da * 60 * 60 * 24 / 100
        if model in ['IITM-ESM']: 
            da = da * 1000
    if var_name == 'clwvi':
        if model in ['IITM-ESM']: 
            da = da/1000
    return da

def save_in_scratch(var_name, model, experiment, resolution, timescale, source, ds):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}/{source}'
    filename = f'{model}_{var_name}_{timescale}_*_{experiment}_{resolution}.nc'
    if resolution == 'regridded':
        filename = f'{model}_{var_name}_{timescale}_{experiment}_{resolution}_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc'
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(f'{folder}/{filename}')
    ds.to_netcdf(path, mode="w")
    print(f'{model} {var_name} data saved at {path}')
    return path

def process_data(var_name, model, experiment, resolution, timescale, source):
    print(f'Processing {model} {resolution} {timescale} {var_name} data from {experiment} experiment')
    if resolution == 'regridded':
        print(f'Regridding to {cD.x_res}x{cD.y_res} degrees')
    da = get_var_data(var_name, model, experiment, resolution, timescale, source)[var_name]
    da = convert_units(model, da, var_name)
    ds = xr.Dataset({var_name: da}) if not var_name == 'ds_cl' else da
    path = save_in_scratch(var_name, model, experiment, resolution, timescale, source, ds)
    return xr.open_dataset(path, chunks= {'time': 'auto'})[var_name]


# --------------------------------------------------------------------------------- request variable ----------------------------------------------------------------------------------------------------- #
def request_process(var_name, model, experiment, resolution, timescale, source, path):
    print(f'no {model} {experiment} {resolution} {timescale} {var_name} data at {cD.x_res}x{cD.y_res} deg in \n {path}')
    response = input(f"Do you want to process {var_name} from {model}? (y/n/y_all) (check folder first): ").lower()
    # response = 'y'
    if response == 'y':
        da = process_data(var_name, model, experiment, resolution, timescale, source)
        print('requested dataset is processed and saved in scratch')
    if response == 'n':
        print('exiting')
        exit()
    if response == 'y_all':
        for model, experiment in mD.run_dataset(var_name, cD.datasets, cD.experiments):
            process_data(var_name, model, experiment, resolution, timescale, source)
            print('all requested datasets processed and saved in scratch')
        exit()
    return da

def check_scratch(var_name, model, experiment, resolution, timescale, source):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}/{source}'
    filename = f'{model}_{var_name}_{timescale}_{experiment}_{resolution}.nc'
    if resolution == 'regridded':
        filename = f'{model}_{var_name}_{timescale}_{experiment}_{resolution}_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc'
    path = f'{folder}/{filename}'
    return path, os.path.exists(path)

def pick_ocean_region(da):
    mask = xr.open_dataset('/home/565/cb4968/Documents/code/phd/util-data/cmip/ocean_mask.nc')['ocean_mask']
    da = da * mask
    return da

def get_cmip_data(switch_var = {'pr': True}, switch = {'test_sample': False, 'ocean_mask': False}, model = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    source = find_source(model)
    var_name = next((key for key, value in switch_var.items() if value), None)
    path, in_scratch = check_scratch(var_name, model, experiment, resolution, timescale, source)
    if in_scratch:
        ds = xr.open_dataset(path, chunks= {'time': 'auto'}) 
        da = ds[var_name]
        if switch.get('test_sample', False):
            da = da.isel(time = slice(0, 365))  if switch['test_sample']   else da
        if switch.get('ocean_mask', False):
            da = pick_ocean_region(da)          if switch['ocean_mask']    else da
        return da
    else:
        da = request_process(var_name, model, experiment, resolution, timescale, source, path) # process the data and save in scratch
        return da



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':

    switch_var = {
        'pr':       False,                                                                                      # Precipitation
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      False,  'hus' :         False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     True,                                          # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,                      # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Geopotential
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {
        'test_sample':   False, # first year (or shorter)
        'ocean_mask':    False, # mask
        }

    switch_test = {
        'test_something':           True,
        'check NaN':                True,
        'plot':                     False,
        'delete_previous_plots':    True,
        'show_plots':               False
        }


    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import map_plot as mP
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
    
    # ----------------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #    
    if switch_test['test_something']:
        var_name = next((key for key, value in switch_var.items() if value), None)
        for model, experiment in mD.run_dataset(var_name, cD.datasets, cD.experiments):
            da = get_cmip_data(switch_var, switch, model = model, experiment = experiment, timescale = cD.timescales[0])

        # --------------------------------------------------------------------------------- Check NaN, min, max ----------------------------------------------------------------------------------------------------- #
            if switch_test['check NaN']:
                nan_count = np.sum(np.isnan(da))
                print(f'Number of NaN: {nan_count.compute().data}')
                print(f'min value: {np.min(da).compute().data}')
                print(f'max value: {np.max(da).compute().data}')

        # ------------------------------------------------------------------------------------- Plot fields ----------------------------------------------------------------------------------------------------- #
            if switch_test['plot']:
                ds = xr.Dataset()
                ds[f'{model}-{experiment}'] = da.isel(time=0)
                filename = f'{var_name}_{model}-{experiment}.png'
                vmin = None
                vmax = None
                label = 'units []'
                fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = 'Blues', variable_list = list(ds.data_vars.keys()))
                mP.show_plot(fig, show_type = 'save_cwd', filename = filename)





