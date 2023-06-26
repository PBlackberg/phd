import numpy as np
import xarray as xr
import os

# -------------------------------------------------------------------------------------- For choosing dataset / model----------------------------------------------------------------------------------------------------- #

models_cmip5 = [
    # 'IPSL-CM5A-MR', # 1
    # 'GFDL-CM3',     # 2
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

models_cmip6 = [
    'TaiESM1',        # 1
    # 'BCC-CSM2-MR',    # 2
    # 'FGOALS-g3',      # 3
    # 'CNRM-CM6-1',     # 4
    # 'MIROC6',         # 5
    # 'MPI-ESM1-2-LR',  # 6
    # 'NorESM2-MM',     # 7
    # 'GFDL-CM4',       # 8
    # 'CanESM5',        # 9
    # 'CMCC-ESM2',      # 10
    # 'UKESM1-0-LL',    # 11
    # 'MRI-ESM2-0',     # 12
    # 'CESM2',          # 13
    # 'NESM3'           # 14
    ]

observations = [
    # 'GPCP',           # precipitation
    # 'ISCCP'             # clouds (weather states)
    ]

datasets = models_cmip5 + models_cmip6 + observations

experiments = [
    'historical',
    # 'rcp85',          # warm scenario cmip5
    # 'ssp585',         # warm scenario for cmip6
    # ''                  # observations
    ]

timescales = [
    # 'daily',
    'monthly'
    ]

resolutions = [
    # 'orig',
    'regridded'
    ]




# -------------------------------------------------------------------------------------- Determining folder to save to ----------------------------------------------------------------------------------------------------- #

folder_save = f'{os.path.expanduser("~")}/Documents/data'
folder_save_gadi = '/g/data/k10/cb4968/data'

# --------------------
# structure of folders 
# for metric: folder_save/variable_type/metrics/metric/source/dataset_filename ex: pr/metrics/rxday/cmip6/FGOALS-g3_rxday_historical_regridded.nc
# for figure: folder_save/variable_type/figures/plot_metric/source/source_filename ex: pr/figures/rxday_tMean/cmip6/cmip6_rx1day_regridded.pdf 
# --------------------





# ------------------------------------------------------------------------------------------------ save / load ----------------------------------------------------------------------------------------------------- #

def save_file(data, folder, filename):
    ''' Saves file to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + filename
    if os.path.exists(path):
        os.remove(path)    
    data.to_netcdf(path)
    return

def save_figure(figure, folder, filename):
    ''' Save figure to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    if os.path.exists(path):
        os.remove(path)    
    figure.savefig(path)
    return

def save_sample_data(data, folder_save, source, dataset, name, timescale='monthly', experiment='historical', resolution='regridded'):
    ''' Save sample data (gadi) '''
    folder = f'{folder_save}/sample_data/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{dataset}_{name}_{timescale}_{experiment}_{resolution}.nc'
    save_file(data, folder, filename)
    return

def save_metric(data, folder_save, metric, source, dataset, experiment='historical', resolution='regridded'):
    ''' Save calculated metric to file '''
    folder = f'{folder_save}/metrics/{metric}/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{dataset}_{metric}_{experiment}_{resolution}.nc' if experiment else f'{dataset}_{metric}_{resolution}.nc'
    save_file(data, folder, filename)
    return

def save_metric_figure(figure, folder_save, metric, source, name, resolution='regridded'):
    ''' Save plot of metric calculation to file '''
    folder = f'{folder_save}/figures/{metric}/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{name}_{resolution}.pdf'
    save_figure(figure, folder, filename)
    return None

def load_sample_data(folder_load, dataset, name, timescale='monthly', experiment='historical', resolution='regridded'):
    ''' Load saved sample data'''
    data_sources = ['cmip5', 'cmip6', 'obs']
    for source in data_sources:
        folder = f'{folder_load}/sample_data/{source}'
        filename = f'{dataset}_{name}_{timescale}_{experiment}_{resolution}.nc'
        file_path = os.path.join(folder, filename)
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except FileNotFoundError:
            continue
    print(f'Error: no file at ex: {file_path}')
    return None

def load_metric(folder_load, variable_type, metric, dataset, experiment='historical', resolution='regridded'):
    ''' Load metric data '''
    data_sources = ['cmip5', 'cmip6', 'obs']
    for source in data_sources:
        folder = f'{folder_load}/{variable_type}/metrics/{metric}/{source}'
        filename = f'{dataset}_{metric}_{experiment}_{resolution}.nc'
        file_path = os.path.join(folder, filename)
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except FileNotFoundError:
            continue
    print(f"Error: no file found for {dataset} - {metric}, example: {file_path}")
    return None




# ---------------------------------------------------------------------------------------- Other variables / functions ----------------------------------------------------------------------------------------------------- #

def find_source(dataset, models_cmip5, models_cmip6, observations):
    '''Determining source of dataset '''
    if np.isin(models_cmip5, dataset).any():
        source = 'cmip5' 
    elif np.isin(models_cmip6, dataset).any():
        source = 'cmip6' 
    elif np.isin(observations, dataset).any():
        source = 'obs' 
    else:
        source = 'test' 
    return source

def find_list_source(datasets, models_cmip5, models_cmip6, observations):
    ''' Determining source of dataset list '''
    sources = set()
    for dataset in datasets:
        sources.add('cmip5') if dataset in models_cmip5 else None
        sources.add('cmip6') if dataset in models_cmip6 else None
        sources.add('obs') if dataset in observations else None
    if   'cmip5' in sources and 'cmip6' in sources:
         return 'mixed'
    elif 'cmip5' in sources:
         return 'cmip5'
    elif 'cmip6' in sources:
         return 'cmip6'
    else:
         return 'obs'

def find_ifWithObs(datasets, observations):
    ''' Indicate if there is observations in the dataset list (for filename of figures) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''

def data_exist(model, experiment):
    ''' Check if model/project has data
    (for precipitation a model is not included if it does not have daily precipitation data)
    '''
    data_exist = 'True'
    return data_exist


def no_data(source, experiment, data_exists):
    if experiment and source in ['cmip5', 'cmip6']:
        pass
    elif not experiment and source == 'obs':
        pass
    else:
        return True

    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']:
        return True

    if not data_exists:
        return True
    

institutes_cmip5 = {
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


institutes_cmip6 = {
    'TaiESM1':'AS-RCEC',
    'BCC-CSM2-MR':'BCC',
    'FGOALS-g3':'CAS',
    'CNRM-CM6-1':'CNRM-CERFACS',
    'MIROC6':'MIROC',
    'MPI-ESM1-2-LR':'MPI-M',
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


institutes = {**institutes_cmip5, **institutes_cmip6}



# not included from cmip6:
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







