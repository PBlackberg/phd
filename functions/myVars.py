import numpy as np
import xarray as xr
import os

# -------------------------------------------------------------------------------------- For choosing dataset / model----------------------------------------------------------------------------------------------------- #

models_cmip5 = [
    # 'IPSL-CM5A-MR',      # 1
    # 'GFDL-CM3',          # 2
    # 'GISS-E2-H',         # 3
    # 'bcc-csm1-1',        # 4
    # 'CNRM-CM5',          # 5
    # 'CCSM4',             # 6
    # 'HadGEM2-AO',        # 7
    # 'BNU-ESM',           # 8
    # 'EC-EARTH',          # 9
    # 'FGOALS-g2',         # 10
    # 'MPI-ESM-MR',        # 11
    # 'CMCC-CM',           # 12
    # 'inmcm4',            # 13
    # 'NorESM1-M',         # 14
    # 'CanESM2',           # 15
    # 'MIROC5',            # 16
    # 'HadGEM2-CC',        # 17
    # 'MRI-CGCM3',         # 18
    # 'CESM1-BGC'          # 19
    ]

models_cmip6 = [
    'TaiESM1',           # 1
    'BCC-CSM2-MR',       # 2
    'FGOALS-g3',         # 3
    'CNRM-CM6-1',        # 4
    'MIROC6',            # 5
    'MPI-ESM1-2-LR',     # 6
    'NorESM2-MM',        # 7
    'GFDL-CM4',          # 8
    'CanESM5',           # 9
    'CMCC-ESM2',         # 10
    'UKESM1-0-LL',       # 11
    'MRI-ESM2-0',        # 12
    'CESM2-WACCM',       # 19
    'NESM3',             # 14
    'IITM-ESM',          # 15 (new from here)
    'EC-Earth3',         # 16
    'INM-CM5-0',         # 17
    'IPSL-CM6A-LR',      # 18
    'KIOST-ESM',         # 19
    ]

observations = [
    'GPCP',              # precipitation (from project al33 on gadi)
    # 'ISCCP'              # clouds (weather states) (https://isccp.giss.nasa.gov/wstates/hggws.html)
    # 'CERES'              # radiation (https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp#)
    ]

datasets = models_cmip5 + models_cmip6 + observations

experiments = [
    'historical',     
    # 'rcp85',             # warm scenario cmip5
    # 'ssp585',            # warm scenario for cmip6
    # ''                   # observations
    ]

timescales = [
    'daily',
    # 'monthly'
    ]

resolutions = [
    # 'orig',
    'regridded'
    ]

folder_save = [
    os.path.expanduser("~") + '/Documents/data',
    # '/g/data/k10/cb4968/data'
    ]



# ------------------------------------------------------------------------------------------------ save / load ----------------------------------------------------------------------------------------------------- #

# --------------------
# structure of folders 
# for metric: [folder_save]/[variable_type]/metrics/[metric]/[source]/[dataset]_[filename] ex: [folder_save]/pr/metrics/rxday/cmip6/[filename]
# for figure: [folder_save]/[variable_type]/figures/[plot_metric]/[source]/[source]_[filename] ex: [folder_save]/pr/figures/rxday_tMean/cmip6/[filename]

# structure of filename
# for metric: [dataset]_[metric]_[timescale]_[experiment]_[resolution] ex: FGOALS-g3_rxday_daily_historical_regridded.nc
# for figure_metric: [source]_[metric]_[timescale]_[resolution]  ex: cmip6_rx1day_daily_regridded.pdf 
# --------------------

def save_file(data, folder, filename):
    ''' Saves file to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    os.remove(path) if os.path.exists(path) else None
    data.to_netcdf(path)
    return

def save_figure(figure, folder, filename):
    ''' Save figure to specified folder and filename '''
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)
    return

def save_sample_data(data, folder_save, source, dataset, name, timescale, experiment, resolution):
    ''' Save sample data (gadi) '''
    folder = f'{folder_save}/sample_data/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{dataset}_{name}_{timescale}_{experiment}_{resolution}.nc' if not source == 'obs' else f'{dataset}_{name}_{timescale}_{resolution}.nc'
    save_file(data, folder, filename)
    return

def save_metric(data, folder_save, metric, source, dataset, timescale, experiment, resolution):
    ''' Save calculated metric to file '''
    folder = f'{folder_save}/metrics/{metric}/{source}'
    os.makedirs(folder, exist_ok=True)
    filename = f'{dataset}_{metric}_{timescale}_{experiment}_{resolution}.nc' if not source == 'obs' else f'{dataset}_{metric}_{timescale}_{resolution}.nc'
    save_file(data, folder, filename)
    return

def save_figure_from_metric(figure, folder_save, metric, source, filename):
    ''' Save plot of metric calculation to file '''
    folder = f'{folder_save}/figures/{metric}/{source}'
    save_figure(figure, folder, filename)
    return None

def load_sample_data(folder_load, source, dataset, name, timescale, experiment, resolution):
    ''' Load saved sample data'''
    folder = f'{folder_load}/sample_data/{source}'
    filename = f'{dataset}_{name}_{timescale}_{experiment}_{resolution}.nc' if not source == 'obs' else f'{dataset}_{name}_{timescale}_{resolution}.nc'
    file_path = os.path.join(folder, filename)
    ds = xr.open_dataset(file_path)
    return ds

def load_metric(folder_load, variable_type, metric, source, dataset, timescale, experiment, resolution):
    ''' Load metric data '''
    folder = f'{folder_load}/{variable_type}/metrics/{metric}/{source}'
    filename = f'{dataset}_{metric}_{timescale}_{experiment}_{resolution}.nc' if not source == 'obs' else f'{dataset}_{metric}_{timescale}_{resolution}.nc'
    file_path = os.path.join(folder, filename)
    ds = xr.open_dataset(file_path)
    return ds



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


# ------------------------------------------------------------------------------------------------ load_metrics ----------------------------------------------------------------------------------------------------- #

def get_super(x):
    ''' For adding superscripts in strings (input is string) '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)

def name_region(switch):
    if switch['descent']:
        region = '_d' 
    elif switch['ascent']:
        region = '_a' 
    else:
        region = ''
    return region

def find_general_metric_and_specify_cbar(switch):
    if switch['pr'] or switch['percentiles_pr'] or switch['rx1day_pr'] or switch['rx5day_pr']:
        variable_type = 'pr'
        cmap = 'Blues'
        cbar_label = 'pr [mm day{}]'.format(get_super('-1'))
    if switch['pr']:
        metric = 'pr' 
        metric_option = metric
    if switch['percentiles_pr']:
        metric = 'percentiles_pr' 
        metric_option = 'pr97' # there is also pr95, pr99
    return variable_type, metric, metric_option, cmap, cbar_label






# ---------------------------------------------------------------------------------------- Other variables ----------------------------------------------------------------------------------------------------- #

institutes_cmip5 = {
    'IPSL-CM5A-MR': 'IPSL',
    'GFDL-CM3':     'NOAA-GFDL',
    'GISS-E2-H':    'NASA-GISS',
    'bcc-csm1-1':   'BCC',
    'CNRM-CM5':     'CNRM-CERFACS',
    'CCSM4':        'NCAR',
    'HadGEM2-AO':   'NIMR-KMA',
    'BNU-ESM':      'BNU',
    'EC-EARTH':     'ICHEC',
    'FGOALS-g2':    'LASG-CESS',
    'MPI-ESM-MR':   'MPI-M',
    'CMCC-CM':      'CMCC',
    'inmcm4':       'INM',
    'NorESM1-M':    'NCC',
    'CanESM2':      'CCCma',
    'MIROC5':       'MIROC',
    'HadGEM2-CC':   'MOHC',
    'MRI-CGCM3':    'MRI',
    'CESM1-BGC':    'NSF-DOE-NCAR'
    }

institutes_cmip6 = {
    'TaiESM1':           'AS-RCEC',
    'BCC-CSM2-MR':       'BCC',
    'FGOALS-g3':         'CAS',
    'CanESM5':           'CCCma',
    'CMCC-ESM2':         'CMCC',
    'CNRM-CM6-1':        'CNRM-CERFACS',
    'MIROC6':            'MIROC',
    'MPI-ESM1-2-LR':     'MPI-M',
    'GISS-E2-1-H':       'NASA-GISS',
    'NorESM2-MM':        'NCC',
    'GFDL-CM4':          'NOAA-GFDL',
    'UKESM1-0-LL':       'MOHC',
    'MRI-ESM2-0':        'MRI',
    'CESM2-WACCM':       'NCAR'
    'NESM3':             'NUIST',
    'IITM-ESM':          'CCCR-IITM',
    'EC-Earth3':         'EC-Earth-Consortium',
    'INM-CM5-0':         'INM',
    'IPSL-CM6A-LR':      'IPSL',
    'KIOST-ESM':         'KIOST',
    }

# not included from cmip6:
# 'KACE-1-0-G':      'NIMS-KMA'             (this institute has data for UKESM1-0-LL which is already included from a different institute)
# 'GISS-E2-1-H':     'NASA-GISS'            (only monthly for all variables)
# 'MCM-UA-1-0':      'UA',                  (only monthly data)
# 'CIESM':           'THU',                 (only monthly data)
# 'AWI-CM-1-1-MR':   'AWI'                  (no daily precip)
# 'CAMS-CSM1-0':     'CAMS',                (hardly any other variables than precip daily)
# 'E3SM-1-0':        'E3SM-Project',        (not correct years)
# 'FIO-ESM-2-0':     'FIO-QLNM',            (only monthly)
# 'MPI-ESM-1-2-HAM': 'HAMMOZ-Consortium'    (no future scenario)
# 'SAM0-UNICON':     'SNU',                 (no future scenario)
# *'CESM2':           'NCAR',                (regular CESM2 does not have monthly hur in ssp585, but does have it in CESM2-WACCM)
institutes = {**institutes_cmip5, **institutes_cmip6}



