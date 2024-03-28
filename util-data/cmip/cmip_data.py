''' 
# ------------------------
#       Get data
# ------------------------
This script loads cmip data from nci, remaps to common grid, and converts units if needed
Function:
    get_cmip_data(switch_var = {'pr': True}, switch = {'test_sample': False}, model = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0])

Input:
    da_cmip - cmip5 or cmip6 data   dim: (time, (plev), lat, lon)       get from: util-data/cmip/cmip_data.py
    da_obs  - observational data    dim: (time, (plev), lat, lon)       get from: util-data/observations/obs_data.py
    da_icon - icon model data       dim: (time, (plev), lat, lon)       get from: util-data/icon/icon_data.py

Output:
    da: - data array                dim: (time, (plev), lat, lon) 
'''


# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import warnings
from pathlib import Path


# --------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
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
#     Handle folders
# ------------------------
# -------------------------------------------------------------------------------------- get folder ----------------------------------------------------------------------------------------------------------#
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


# ----------------------------------------------------------------------------------- open files ----------------------------------------------------------------------------------------------------------#
def concat_data(path_folder, experiment):
    ''' Concatenate files between specified years '''
    files = [f for f in os.listdir(path_folder) if f.endswith('.nc')]
    if experiment == 'historical':
        year1 = cD.cmip_years[0][0].split('-')[0]
        year2 = cD.cmip_years[0][0].split('-')[1]
    else:
        year1 = cD.cmip_years[0][1].split('-')[0]
        year2 = cD.cmip_years[0][1].split('-')[1]
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



# ------------------------
#   Remap coordinates
# ------------------------
# --------------------------------------------------------------------------------- remap ----------------------------------------------------------------------------------------------------------#
def regrid_hor(ds, da): 
    ''' Regrid to common horizontal grid (ds and da from the same model)'''
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="esmpy")
    regridder = rGh.regrid_conserv_xesmf(ds) # define regridder based of grid from other model (FGOALS-g2 from cmip5 currently)
    warnings.resetwarnings()
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
# ----------------------------------------------------------------------------- get cloud fraction variable ----------------------------------------------------------------------------------------------------------#
def get_cmip_cl_data(var_name, model, experiment, resolution, timescale, source, ensemble, project, time_interval): 
    ''' For hybrid-sigma pressure coordinates interpolation to pressure coordinates '''       
    if var_name in ['cl']:                                  # interpolation takes a long time, so stored locally       
        path = f'/g/data/k10/cb4968/data/sample_data/cl/{source}/{model}_cl_{timescale}_{experiment}_{resolution}.nc'                                                                                            
        return xr.open_dataset(path) #['cl']         
    else:                                                   # for interpolation in 'cl_vert_interp' in util folder (could extend to other variables on hybrid-sigma pressure coordinates)
        print('need to fix function first')
        # exit()
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

def get_var_data(var_name, model, experiment, resolution, timescale, source):
    ''' concatenate files, interpolate grid, and mask '''
    ensemble = folder_ensemble(source, model, experiment)
    project = 'CMIP'        if experiment == 'historical'   else 'ScenarioMIP'
    time_interval = 'day'   if timescale == 'daily'         else 'Amon'         
    if var_name in ['cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid']:                                                  # cloud variable is on hybrid-sigma pressure coordiantes, so treated separately
        return get_cmip_cl_data(var_name, model, experiment, resolution, timescale, source, ensemble, project, time_interval)
    else:                                                                                                       # all other 2D or 3D pressure level variables     
        folder_to_var = folder_var(var_name, model, experiment, ensemble, project, time_interval)
        # path_folder = '/g/data/oi10/replicas/CMIP6/CMIP/EC-Earth-Consortium/EC-Earth3/historical/r11i1p1f1/Amon/clwvi/gr/v20200201/'
        ds = concat_data(folder_to_var, experiment)    
        print('concatenated files')
        da = ds[var_name]      
        da = regrid_vert(da, model)                         if 'plev' in da.dims                 else da    # vertically interpolate (some models have different number of vertical levels)
        da = regrid_hor(ds, da)                             if resolution == 'regridded'         else da    # horizontally interpolate
        print('regridded')
    return xr.Dataset(data_vars = {f'{var_name}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)             # for when running original resolution. If regridded it should already be lat: [-30,30]

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

def process_data(var_name, model, experiment, resolution, timescale, source):
    print(f'Processing {model} {resolution} {timescale} {var_name} data from {experiment} experiment')
    if resolution == 'regridded':
        print(f'Regridding to {cD.x_res}x{cD.y_res} degrees')
    da = get_var_data(var_name, model, experiment, resolution, timescale, source)[var_name]
    da = convert_units(model, da, var_name)
    ds = xr.Dataset({var_name: da}) if not var_name == 'ds_cl' else da
    print('finished process')
    return ds

def folder_structure(var_name, source):
    folder = f'{sF.folder_scratch}/sample_data/{var_name}/{source}'
    # print(folder)
    return folder

def filename_structure(model, experiment, var_name, source, timescale = cD.timescales[0]):
    # print(var_name)
    if source in ['cmip5', 'cmip6'] and experiment == 'historical':
        filename = f'{model}_{var_name}_{timescale}_{experiment}_{cD.cmip_years[0][0]}_{cD.resolutions[0]}'
    if source in ['cmip5', 'cmip6'] and experiment in ['ssp585', 'rcp85']:
        filename = f'{model}_{var_name}_{timescale}_{experiment}_{cD.cmip_years[0][1]}_{cD.resolutions[0]}'
    if cD.resolutions[0] == 'regridded':
        filename = f'{filename}_{int(360/cD.x_res)}x{int(180/cD.y_res)}'
    # print(f'filename is: {filename}')
    return filename

def save_in_scratch(var_name, model, experiment, resolution, timescale, source, ds):
    folder = folder_structure(var_name, source)
    filename = filename_structure(model, experiment, var_name, source, timescale = cD.timescales[0])
    Path(folder).mkdir(parents=True, exist_ok=True)
    path = Path(f'{folder}/{filename}.nc')
    ds.to_netcdf(path, mode="w")
    print(f'{model} {var_name} data saved in scratch at {path}')
    return path

def request_process(var_name, model, experiment, resolution, timescale, source, path):
    print(f'no {model} {experiment} {resolution} {timescale} {var_name} data at {cD.x_res}x{cD.y_res} deg in \n {path}')
    response = input(f"Do you want to process {var_name} from {model}? (y/n) (check folder first): ").lower()
    # response = 'y'
    if response == 'y':
        ds = process_data(var_name, model, experiment, resolution, timescale, source)
    if response == 'n':
        print('exiting')
        exit()
    return ds



# ----------------------------
#     handle data request
# ----------------------------
def check_if_in_scratch(var_name, model, experiment, resolution, timescale, source):
    folder = folder_structure(var_name, source)
    filename = filename_structure(model, experiment, var_name, source, timescale = cD.timescales[0])
    path = f'{folder}/{filename}.nc'
    return path, os.path.exists(path)

def get_data_from_folder(switch, var_name, model, experiment, resolution, timescale, source):
    if switch.get('re_process', False):
        ds = process_data(var_name, model, experiment, resolution, timescale, source)
        path = save_in_scratch(var_name, model, experiment, resolution, timescale, source, ds)
    path, in_scratch = check_if_in_scratch(var_name, model, experiment, resolution, timescale, source)
    if in_scratch:
        ds = xr.open_dataset(path, chunks= {'time': 'auto'}) 
        return ds
    else:
        ds = request_process(var_name, model, experiment, resolution, timescale, source, path)
        path = save_in_scratch(var_name, model, experiment, resolution, timescale, source, ds)
        return ds

def get_cmip_data(switch_var = {'pr': True}, switch = {'test_sample': False}, model = '', experiment = '', 
                  resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    source = find_source(model)
    var_name = next((key for key, value in switch_var.items() if value), None)
    if switch.get('from_scratch', False):
        ds = get_data_from_folder(switch, var_name, model, experiment, resolution, timescale, source)
    else:
        ds = process_data(var_name, model, experiment, resolution, timescale, source)
    if switch.get('test_sample', False):
        ds = ds.isel(time = slice(0, 365))
    return ds[var_name]    



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot        as mP
    import get_plot.show_plots      as sP


    switch_var = {
        'pr':       False,                                                                                      # Precipitation
        'tas':      False, 'ta':            False,                                                              # Temperature
        'wap':      False,                                                                                      # Circulation
        'hur':      False, 'hus' :          False,                                                              # Humidity                   
        'rlds':     False, 'rlus':          False,  'rlut':     False,                                          # Longwave radiation
        'rsdt':     False, 'rsds':          False,  'rsus':     False,  'rsut':     False,                      # Shortwave radiation
        'cl':       False, 'cl_p_hybrid':   False,  'p_hybrid': False,  'ds_cl':    False,                      # Cloudfraction (ds_cl is for getting pressure levels)
        'zg':       False,                                                                                      # Geopotential
        'hfss':     False, 'hfls':          False,                                                              # Surface fluxes
        'clwvi':    False,                                                                                      # Cloud ice and liquid water
        }

    switch = {
        'test_sample':   False,                                                                                 # first year
        'from_scratch':  True,   're_process':   True,  # if both are true the scratch file is replaced with the reprocessed version
        }

    switch_test = {
        'delete_test_plots':        False,
        'plot_scene':               True,
        'check NaN':                False,
        }

    sP.remove_test_plots() if switch_test['delete_test_plots'] else None
    

    # ------------------------------------------------------------------------ Run to test / save variable temporarily --------------------------------------------------------------------------------------------------- #
    var_name = next((key for key, value in switch_var.items() if value), None)
    print(f'variable: {var_name}')
    for experiment in cD.experiments:
        print(f'experiment: {experiment}')
        for model in mD.run_dataset_only(var_name, cD.datasets):
            print(f'dataset: {model}')
            # -------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #    
            da = get_cmip_data(switch_var, switch, model = model, experiment = experiment, timescale = cD.timescales[0])
            da.load()
            
            # print(da)
            # exit()
            

            ds = xr.Dataset()
            # -------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #    
            if switch_test['plot_scene']:   # intended for running one variable and several datasets
                if 'plev' in da.dims:
                    level = 500e2
                    ds[model] = da.isel(time = 0).sel(plev = level)
                else:
                    ds[model] = da.isel(time = 0)

            if switch_test['check NaN']:
                nan_count = np.sum(np.isnan(da))
                print(f'Number of NaN: {nan_count.compute().data}')
                print(f'min value: {np.min(da).compute().data}')
                print(f'max value: {np.max(da).compute().data}')


    # ------------------------------------------------------------------------------------------- Plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_scene']:           # plots several datasets with shared colormap
        label = '[units]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'cmip_{experiment}_{var_name}.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()), cat_cmap = False)
        sP.show_plot(fig, show_type = 'save_cwd', filename = filename)



