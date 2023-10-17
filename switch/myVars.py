import os
import numpy as np



# ------------------------
#    Choose datasets
# ------------------------
# ---------------------------------------------------------------- Constructed / Random fields ----------------------------------------------------------------------------------------------------- #
constructed_fields = [
    # 'constructed'        # 1
    # 'random'             # 2
    ]


# -------------------------------------------------------------------------  CMIP5 ----------------------------------------------------------------------------------------------------- #
models_cmip5 = [
    # 'IPSL-CM5A-MR',    # 1
    # 'GFDL-CM3',        # 2
    # 'GISS-E2-H',       # 3
    # 'bcc-csm1-1',      # 4
    # 'CNRM-CM5',        # 5
    # 'CCSM4',           # 6
    # 'HadGEM2-AO',      # 7
    # 'BNU-ESM',         # 8
    # 'EC-EARTH',        # 9
    # 'FGOALS-g2',       # 10
    # 'MPI-ESM-MR',      # 11
    # 'CMCC-CM',         # 12
    # 'inmcm4',          # 13
    # 'NorESM1-M',       # 14
    # 'CanESM2',         # 15
    # 'MIROC5',          # 16
    # 'HadGEM2-CC',      # 17
    # 'MRI-CGCM3',       # 18
    # 'CESM1-BGC'        # 19
    ]


# -------------------------------------------------------------------------  CMIP6 ----------------------------------------------------------------------------------------------------- #
models_cmip6 = [         # Models ordered by change in temperature with warming (similar models removed)
    # 'INM-CM5-0',         # 1
    # 'IITM-ESM',          # 2
    # 'FGOALS-g3',         # 3                                 
    # 'MIROC6',            # 4                                      
    # 'MPI-ESM1-2-LR',     # 5                                      
    'KIOST-ESM',         # 6 
    # 'BCC-CSM2-MR',       # 7        
    # 'GFDL-ESM4',         # 8        
    # 'NorESM2-LM',        # 9
    # 'NorESM2-MM',        # 10                                      
    # 'MRI-ESM2-0',        # 11                                  
    # 'GFDL-CM4',          # 12      
    # 'CMCC-CM2-SR5',      # 13                  
    # 'CMCC-ESM2',         # 14                                    
    # 'NESM3',             # 15     
    # 'ACCESS-ESM1-5',     # 16 
    # 'CNRM-ESM2-1',       # 17  
    # 'EC-Earth3',         # 18
    # 'CNRM-CM6-1',        # 19                       
    # 'IPSL-CM6A-LR',      # 20
    # 'ACCESS-CM2',        # 21
    # 'TaiESM1',           # 22                                      
    # 'CESM2-WACCM',       # 23   
    # 'CanESM5',           # 24
    # 'UKESM1-0-LL',       # 25
    # 'CNRM-CM6-1-HR',     # 26
    # 'INM-CM4-8',         # 27
    # 'MIROC-ES2L',        # 28
    # 'KACE-1-0-G',        # 29
    ]

switch_exclude = {              # not mutually exclusive (10 models removed from: simliar_versions, no_clouds, and no high_res)
    'highlight':                False,
    'no_similar_versions':      False,
    'no_clouds':                False,
    'no_high_res_version':      False,

    'no_ocean':                 False,
    'no_ta':                    False,
    'not_in_schiro':            False,
    'only_res_versions':        False,
    }


# -----------------------------------------------------------------------  Observations ----------------------------------------------------------------------------------------------------- #
observations = [
    # 'GPCP',              # for precipitation and organization index (from project al33 on gadi)
    # 'GPCP_1998-2009',    # high offset in high percentile precipitation
    # 'GPCP_2010-2022',    # low offset in high percentile precipitation
    # 'ISCCP',             # clouds (weather states) (https://isccp.giss.nasa.gov/wstates/hggws.html) (2000-01 2017-12)
    # 'CERES',             # radiation (https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp#) (2000-03-2023-04)
    # 'ERA5'               # humidity (from project rt52 on gadi) (1998-01 - 2021-12)
    ]


# ------------------------------------------------------------------------ Overall settings ----------------------------------------------------------------------------------------------------- #
timescales = [
    # 'daily',
    'monthly',
    # 'annual'
    ]

experiments = [
    # 'historical',     
    # 'rcp85',             # warm scenario for cmip5
    'ssp585',              # warm scenario for cmip6
    # ''                   # observations
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


# ------------------------------------------------------------------------ Folder to save metric to ----------------------------------------------------------------------------------------------------- #
folder_save = [
    os.path.expanduser("~") + '/Documents/data',
    # '/g/data/k10/cb4968/data'
    ]



# --------------------------------
#  Functions for picking datasets
# --------------------------------
# -------------------------------------------------------------------- Deal with missing data ----------------------------------------------------------------------------------------------------- #
def exclude_models(models_cmip6, switch_exclude):
    ''' Some models are versions of the same model and give close to identical results. Some models are exluded to fit into plots.'''
    models_excluded = []
    if switch_exclude['no_similar_versions']:
        m1, m2, m3 =         'GFDL-ESM4', 'CMCC-CM2-SR5', 'CNRM-ESM2-1' # these models are very similar in time-mean
        models_cmip6 = list(filter(lambda x: x not in (m1, m2, m3), models_cmip6))
        models_excluded.extend([m for m in            (m1, m2, m3) if m not in models_excluded])
    
    if switch_exclude['no_clouds']:
        m1, m2, m3, m4, m5 = 'INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'UKESM1-0-LL', 'INM-CM4-8' # calculation of cloud variable was just different in UK model
        models_cmip6 = list(filter(lambda x: x not in (m1, m2, m3, m4, m5), models_cmip6))
        models_excluded.extend([m for m in            (m1, m2, m3, m4, m5) if m not in models_excluded])

    if switch_exclude['no_high_res_version']:
        m1, m2 =             'NorESM2-MM', 'CNRM-CM6-1-HR'
        models_cmip6 = list(filter(lambda x: x not in (m1, m2), models_cmip6))
        models_excluded.extend([m for m in            (m1, m2) if m not in models_excluded])

    if switch_exclude['no_ocean']:
        m1, m2, m3, m4, m5 = 'IITM-ESM', 'BCC-CSM2-MR', 'NESM3', 'UKESM1-0-LL', 'CNRM-ESM2-1' # no ocean mask on original grid
        models_cmip6 = list(filter(lambda x: x not in (m1, m2, m3, m4, m5), models_cmip6))
        models_excluded.extend([m for m in            (m1, m2, m3, m4, m5) if m not in models_excluded])

    if switch_exclude['no_ta']:
        m1 =                 'KIOST-ESM'
        models_cmip6 = list(filter(lambda x: x not in (m1), models_cmip6))
        models_excluded.extend([m for m in            (m1) if m not in models_excluded])

    if switch_exclude['not_in_schiro']:
        m1, m2, m3, m4, m5 = ''
        models_cmip6 = list(filter(lambda x: x not in (m1, m2, m3, m4, m5), models_cmip6))
        models_excluded.extend([m for m in            (m1, m2, m3, m4, m5) if m not in models_excluded])

    if switch_exclude['only_res_versions']:
        m1, m2, m3, m4 =     'NorESM2-LM', 'NorESM2-MM', 'CNRM-CM6-1', 'CNRM-CM6-1-HR'
        models_cmip6 = list(filter(lambda x: x not in (m1, m2, m3, m4), models_cmip6))
        models_excluded.extend([m for m in            (m1, m2, m3, m4) if m not in models_excluded])

    return models_cmip6, models_excluded
models_cmip6, models_excluded = exclude_models(models_cmip6, switch_exclude)

def data_available(source = '', dataset = '', experiment = '', var = '', switch = {'ocean_mask': False}):
    ''' Check if dataset has variable '''
    # Invalid source and dataset combination for for-loops
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # only run fitting scenario for cmip version
        return  False
    if not experiment and not source in ['obs', 'test']:                                          # only run obs or other for experiment == ''
        return False
    if experiment and source in ['obs']:                                                          # only run models when experiment ~ '' 
        return False

    # Temperature
    if var in ['ta', 'stability'] and dataset in ['KIOST-ESM']:
        print(f'No {var} data for this dataset')
        return False
    
    # Clouds
    if var in ['lcf', 'hcf'] and dataset in ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'UKESM1-0-LL']:
        print(f'No {var} data for this dataset')
        return False
    
    # Ocean mask
    for mask_type in [k for k, v in switch.items() if v]:
        if mask_type == 'ocean_mask' and dataset in ['IITM-ESM', 'BCC-CSM2-MR', 'NESM3', 'UKESM1-0-LL', 'CNRM-ESM2-1']: 
            print(f'No original grid ocean mask for this dataset')
            return True
    return True


# -------------------------------------------------------------------- Label the source ----------------------------------------------------------------------------------------------------- #
def find_source(dataset, models_cmip5, models_cmip6, observations):
    '''Determining source of dataset '''
    source = 'cmip5' if np.isin(models_cmip5, dataset).any() else None      
    source = 'cmip6' if np.isin(models_cmip6, dataset).any() else source         
    source = 'test'  if np.isin(constructed_fields, dataset).any()        else source     
    source = 'obs'   if np.isin(observations, dataset).any() else source
    return source

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

def get_ds_highlight(switch_highlight, datasets, switch_exclude= {'a':False}, func = exclude_models):
    dataset_highlight = []
    for item in [k for k, v in switch_highlight.items() if v]:
        if  item == 'by_dTas':
            dataset_highlight = datasets[-int(len(datasets)/2):] # Largest increase in dTas
        if  item == 'by_org_hur_corr':
            dataset_highlight = ['MIROC6', 'NorESM2-LM', 'TaiESM1']
        if  item == 'by_obs_sim':
            dataset_highlight = ['MIROC6', 'TaiESM1']
        if  item == 'by_excluded':
            _, dataset_highlight = func(datasets, switch_exclude)
    return dataset_highlight

    # model_highlight = ['MIROC6', 'NorESM2-LM', 'NorESM2-MM', 'CMCC-ESM2', 'ACCESS-ESM1-5', 'CNRM-CM6-1', 'ACCESS-CM2', 'TaiESM1', 'CESM2-WACCM', 'UKESM1-0-LL']                         # hur sensitive to org



# ------------------------
# Other dataset variables
# ------------------------
# ---------------------------------------------------------------------------------------- ECS ----------------------------------------------------------------------------------------------------- #
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


# ---------------------------------------------------------------------------------------- Institute list ----------------------------------------------------------------------------------------------------- #
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
    'FGOALS-f3-L':       'CAS',
    'CanESM5':           'CCCma',
    'CMCC-ESM2':         'CMCC',
    'CMCC-CM2-SR5':      'CMCC',
    'CNRM-CM6-1':        'CNRM-CERFACS',
    'CNRM-CM6-1-HR':     'CNRM-CERFACS',
    'CNRM-ESM2-1':       'CNRM-CERFACS',
    'MIROC6':            'MIROC',
    'MIROC-ES2L':        'MIROC',
    'MPI-ESM1-2-LR':     'MPI-M',
    'GISS-E2-1-H':       'NASA-GISS',
    'GISS-E2-1-G':       'NASA-GISS',
    'NorESM2-MM':        'NCC',
    'NorESM2-LM':        'NCC',
    'GFDL-CM4':          'NOAA-GFDL',
    'GFDL-ESM4':         'NOAA-GFDL',
    'UKESM1-0-LL':       'MOHC',
    'KACE-1-0-G':        'NIMS-KMA',
    'MRI-ESM2-0':        'MRI',
    'CESM2':             'NCAR',
    'CESM2-WACCM':       'NCAR',
    'NESM3':             'NUIST',
    'IITM-ESM':          'CCCR-IITM',
    'EC-Earth3':         'EC-Earth-Consortium',
    'INM-CM5-0':         'INM',
    'INM-CM4-8':         'INM',
    'IPSL-CM6A-LR':      'IPSL',
    'KIOST-ESM':         'KIOST',
    'ACCESS-ESM1-5':     'CSIRO',
    'ACCESS-CM2':        'CSIRO-ARCCSS'
    }


# not included from cmip6:
    # 'AWI-CM-1-1-MR':   'AWI'                  (no daily precipitation)                            # 1
    # 'GISS-E2-1-G',     'NASA-GISS'            (no daily precipitation)                            # 2
    # 'CAS-ESM2-0':      'CAS'                  (no daily precipitation)                            # 3
    # 'CanESM5-1':       'CCCma'                (no daily precipitation)                            # 4
    # 'CanESM5-CanOE':   'CCCma'                (no daily precipitation)                            # 5
    # 'GISS-E2-1-H':     'NASA-GISS'            (no daily precipitation)                            # 6
    # 'MCM-UA-1-0':      'UA',                  (no daily precipitation)                            # 7
    # 'CIESM':           'THU',                 (no daily precipitation)                            # 8
    # 'FIO-ESM-2-0':     'FIO-QLNM',            (no daily precipitation)                            # 9
    # 'UKESM1-1-LL':       'MOHC',              (no daily precipitation)  
    # 'ICON-ESM-LR':     'MPI'                  (no daily precipitation) 
    # 'GISS-E2-1-G-CC':  'NASA-GISS'            (no daily precipitation) 
    # 'GISS-E2-2-G':     'NASA-GISS'            (no daily precipitation) 

    # 'NorCPM1 '         'NCC'                  (No monthly hur)
    # 'E3SM-1-0':        'E3SM-Project',        (not correct years)                                 # 16
    # 'EC-Earth3-LR':    'EC-EARTH-Consortium'  (no historical simulation)
    # 'NorESM1-F':       'NCC'                  (no historical simulation)

# could include for part of analysis
    # 'CAMS-CSM1-0':     'CAMS'                 (hardly any other variables except precip daily)    # 17
    # 'HadGEM3-GC31-LL'  'MOHC'                 (hardly any other variables except precip daily)
    # 'HadGEM3-GC31-MM'  'MOHC'                 (hardly any other variables except precip daily)
    # 'FGOALS-f3-L':     'CAS'                  (only monthly variables in future scenario)         # 14
    # 'CESM2':           'NCAR',                (no monthly hur in future scenario)                 # 15
    # no monthly hur in
    # 'MPI-ESM-1-2-HAM': 'HAMMOZ-Consortium'    (no future scenario)                                # 10
    # 'MPI-ESM1-2-LR':   'MPI-M'                (no future scenario)                                # 11
    # 'SAM0-UNICON':     'SNU',                 (no future scenario)                                # 12
    # 'CMCC-CM2-HR4':    'CMCC'                 (no future scenario)                                # 13
    # 'IPSL-CM5A2-INCA': 'IPSL'                 (no future scenario)                                # 13  

# not sure what the FV2 ending is referring to here (might be the same model essentially)
    # CESM2-WACCM-FV2                    
    # CESM2-FV2 


datasets = models_cmip5 + models_cmip6 + observations + constructed_fields
institutes = {**institutes_cmip5, **institutes_cmip6}





