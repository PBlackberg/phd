import os
import numpy as np

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

# 14 models used in Schiro not [1, 2, 6, 13, 15]
# Models ordered by change in temperature with warming
models_cmip6 = [
    # 'INM-CM5-0',         # 1
    # 'IITM-ESM',          # 2
    # 'FGOALS-g3',         # 3                                      
    # 'MIROC6',            # 4                                      
    # 'MPI-ESM1-2-LR',     # 5                                      
    # 'KIOST-ESM',         # 6
    # 'BCC-CSM2-MR',       # 7                                      
    # 'NorESM2-MM',        # 8                                      
    # 'MRI-ESM2-0',        # 9                                      
    # 'GFDL-CM4',          # 10                                     
    # 'CMCC-ESM2',         # 11                                     
    # 'NESM3',             # 12                                     
    # 'EC-Earth3',         # 13
    # 'CNRM-CM6-1',        # 14                                     
    # 'IPSL-CM6A-LR',      # 15
    # 'TaiESM1',           # 16                                       
    # 'CESM2-WACCM',       # 17    
    # 'CanESM5',           # 18
    # 'UKESM1-0-LL',       # 19              
    ]

other = [
    # 'constructed'        # 1
    # 'random'             # 2
    ]

observations = [
    # 'GPCP',              # for precipitation and organization index (from project al33 on gadi)
    # 'GPCP_1998-2009',    # high offset in high percentile precipitation
    # 'GPCP_2010-2022',    # low offset in high percentile precipitation
    'ISCCP',               # clouds (weather states) (https://isccp.giss.nasa.gov/wstates/hggws.html)
    # 'CERES',             # radiation (https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp#)
    # 'ERA5'               # humidity (from project rt52 on gadi)
    ]

datasets = models_cmip5 + models_cmip6 + observations + other


timescales = [
    'daily',
    # 'monthly',
    ]

experiments = [
    # 'historical',     
    # 'rcp85',             # warm scenario for cmip5
    # 'ssp585',              # warm scenario for cmip6
    ''                   # observations
    ]

resolutions = [
    # 'orig',
    'regridded'
    ]

conv_percentiles = [       # for organization metrics
    # '90',
    '95',
    # '97'
    ]

timescales_plot = [
    # 'daily',
    # 'monthly',
    # 'annual',
    ]

folder_save = [
    os.path.expanduser("~") + '/Documents/data',
    # '/g/data/k10/cb4968/data'
    ]



# ---------------------------------------------------------------------------------------- other variables ----------------------------------------------------------------------------------------------------- #

# ECS taken from supplementary information from:
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085782#pane-pcw-figures
# https://www.nature.com/articles/d41586-022-01192-2 # most taken from here (used the column which had the same value for models existing in both sets)

ecs_cmip5 = {
    'IPSL-CM5A-MR': 0,      
    'GFDL-CM3':     0,          
    'GISS-E2-H':    0,         
    'bcc-csm1-1':   0,        
    'CNRM-CM5':     0,          
    'CCSM4':        0,             
    'HadGEM2-AO':   0,        
    'BNU-ESM':      0,           
    'EC-EARTH':     0,          
    'FGOALS-g2':    0,         
    'MPI-ESM-MR':   0,        
    'CMCC-CM':      0,           
    'inmcm4':       0,            
    'NorESM1-M':    0,         
    'CanESM2':      0,           
    'MIROC5':       0,            
    'HadGEM2-CC':   0,        
    'MRI-CGCM3':    0,         
    'CESM1-BGC':    0          
    }

ecs_cmip6 = {
    'TaiESM1':       4.36,              
    'BCC-CSM2-MR':   3.02,      
    'FGOALS-g3':     2.87,       
    'CNRM-CM6-1':    4.90,        
    'MIROC6':        2.6,
    'MPI-ESM1-2-LR': 3.02,
    'NorESM2-MM':    2.49,        
    'GFDL-CM4':      3.89,          
    'CanESM5':       5.64,           
    'CMCC-ESM2':     3.58,         
    'UKESM1-0-LL':   5.36,       
    'MRI-ESM2-0':    3.13,        
    'CESM2-WACCM':   4.68,       
    'NESM3':         4.72,             
    'IITM-ESM':      2.37,          
    'EC-Earth3':     4.26,         
    'INM-CM5-0':     1.92,         
    'IPSL-CM6A-LR':  4.70,      
    'KIOST-ESM':     3.36,
    'CESM2':         5.15         
    }
ecs_list = {**ecs_cmip5, **ecs_cmip6}


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
    'CESM2':             'NCAR',
    'CESM2-WACCM':       'NCAR',
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
# 'CAMS-CSM1-0':     'CAMS',                (hardly any other variables except precip daily)
# 'E3SM-1-0':        'E3SM-Project',        (not correct years)
# 'FIO-ESM-2-0':     'FIO-QLNM',            (only monthly)
# 'MPI-ESM-1-2-HAM': 'HAMMOZ-Consortium'    (no future scenario)
# 'SAM0-UNICON':     'SNU',                 (no future scenario)
# *'CESM2':           'NCAR',               (regular CESM2 does not have monthly hur in ssp585, but does have it in CESM2-WACCM)
institutes = {**institutes_cmip5, **institutes_cmip6}



# -------------------------------------------------------------------- functions for checking available data ----------------------------------------------------------------------------------------------------- #
def find_source(dataset, models_cmip5, models_cmip6, observations):
    '''Determining source of dataset '''
    source = 'cmip5' if np.isin(models_cmip5, dataset).any() else None      
    source = 'cmip6' if np.isin(models_cmip6, dataset).any() else source         
    source = 'test'  if np.isin(other, dataset).any()        else source     
    source = 'obs'   if np.isin(observations, dataset).any() else source
    return source

def data_available(source = '', dataset = '', experiment = '', var = ''):
    ''' Check if dataset has variable '''
    # dataset experiment combination
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # only run fitting scenario for cmip version
        return False
    if not experiment and not source in ['obs', 'test']:                                          # only run obs or other for experiment == ''
        return False
    if experiment and source in ['obs']:                                                          # only run models when experiment ~ '' 
        return False
    
    # available variables
    if var in ['lcf', 'hcf'] and dataset in ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'UKESM1-0-LL']:
        return False
    return True

def find_list_source(datasets, models_cmip5, models_cmip6, observations):
    ''' Determining source of dataset list (for figures) '''
    sources = set()
    for dataset in datasets:
        sources.add('cmip5') if dataset in models_cmip5 else None
        sources.add('cmip6') if dataset in models_cmip6 else None
        sources.add('obs')   if dataset in observations else None
    list_source = 'cmip5' if 'cmip5' in sources else 'test'
    list_source = 'cmip6' if 'cmip6' in sources else list_source
    list_source = 'obs'   if 'obs'   in sources else list_source
    list_source = 'mixed' if 'cmip5' in sources and 'cmip6' in sources else list_source
    return list_source

def find_ifWithObs(datasets, observations):
    ''' Indicate if there is observations in the dataset list (for figures) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''















