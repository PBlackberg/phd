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
models_cmip6 = [         # Models ordered by change in temperature with warming
    'INM-CM5-0',         # 1
    'IITM-ESM',          # 2
    'FGOALS-g3',         # 3    
    'INM-CM4-8',         # 4                                
    'MIROC6',            # 5                                      
    'MPI-ESM1-2-LR',     # 6                         
    'KIOST-ESM',         # 7    
    'BCC-CSM2-MR',       # 8              
    'GFDL-ESM4',         # 9         
    'MIROC-ES2L',        # 10   
    'NorESM2-LM',        # 11      
    'NorESM2-MM',        # 12                                      
    'MRI-ESM2-0',        # 13                                  
    'GFDL-CM4',          # 14      
    'CMCC-CM2-SR5',      # 15                
    'CMCC-ESM2',         # 16                                    
    'NESM3',             # 17     
    'ACCESS-ESM1-5',     # 18   
    'CNRM-ESM2-1',       # 19   
    'EC-Earth3',         # 20
    'CNRM-CM6-1',        # 21  
    'CNRM-CM6-1-HR',     # 22   
    'KACE-1-0-G',        # 23            
    'IPSL-CM6A-LR',      # 24
    'ACCESS-CM2',        # 25   
    'TaiESM1',           # 26                            
    'CESM2-WACCM',       # 27   
    'CanESM5',           # 28  
    'UKESM1-0-LL',       # 29  
    ]

switch_subset = {
    'exclude':         True,  'only_include':      False,                            
    'similar_version': False, 'high_res_version':  False,
    'no_clouds':       True,
    'no_stability':    False, 'stability_no_land': False,
    'no_orig_ocean':   False,
    'not_in_schiro':   False, 
    'res_versions':    False,
    'threshold_res':   False,                                   # Threshold: 2.5 degrees dlat x dlon
    'unrealistic_org': False,                                   # Threshold: qualitative for noew
    'org_insensitive': False,                                   # Threshold: qualitative for noew
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


# -------------------------------------------------------------------------- DYAMOND ----------------------------------------------------------------------------------------------------- #
models_dyamond = [
    # 'winter',
]




# ------------------------------------------------------------------------ Overall settings ----------------------------------------------------------------------------------------------------- #
timescales = [
    # 'daily',
    'monthly',
    # 'annual'
    ]

experiments = [
    'historical',     
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
folder_save = [os.path.expanduser("~") + '/Documents/data']
folder_save = ['/work/bb1153/b382628/data'] if os.path.expanduser("~") == '/home/b/b382628'  else folder_save
folder_save = ['/g/data/k10/cb4968/data']   if os.path.expanduser("~") == '/home/565/cb4968' else folder_save




# --------------------------------
#  Functions for picking datasets
# --------------------------------
# --------------------------------------------------------------------- Exclude models based on conditions ----------------------------------------------------------------------------------------------------- #
def exclude_models(models_cmip6, switch_subset):
    ''' Some models are versions of the same model and give close to identical results. Some models are exluded to fit into plots.'''
    models_excluded = []
    if switch_subset['similar_version']:
        models_exclude = ['GFDL-ESM4', 'CMCC-CM2-SR5', 'CNRM-ESM2-1']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['high_res_version']:
        models_exclude = ['NorESM2-MM', 'CNRM-CM6-1-HR']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['no_clouds']:
        models_exclude = ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3',  'INM-CM4-8', 'CNRM-CM6-1-HR', 'GFDL-ESM4'] # 'GFDL-ESM4' ps missing from pressure calc, calculation of cloud variable was just different in UK model # 'CNRM-CM6-1-HR' no future scenario
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['no_orig_ocean']:
        models_exclude = ['IITM-ESM', 'BCC-CSM2-MR', 'NESM3', 'UKESM1-0-LL', 'CNRM-ESM2-1'] # no ocean mask on original grid
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['no_stability']:
        models_exclude = ['INM-CM4-8', 'GFDL-CM4', 'CESM2-WACCM', 'KACE-1-0-G'] # jist something with KACE temperature in future scenario
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['stability_no_land']:
        models_exclude = ['FGOALS-g3', 'INM-CM4-8', 'MIROC6', 'GFDL-ESM4', 'MIROC-ES2L', 'MRI-ESM2-0', 'GFDL-CM4', 'ACCESS-ESM1-5', 'CNRM-ESM2-1', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'KACE-1-0-G', 'IPSL-CM6A-LR', 'ACCESS-CM2', 'CESM2-WACCM', 'UKESM1-0-LL']  # do not have land values
        # models_exclude = ['INM-CM5-0', 'IITM-ESM', 'MPI-ESM1-2-LR', 'KIOST-ESM', 'BCC-CSM2-MR', 'NorESM2-LM', 'NorESM2-MM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'NESM3', 'EC-Earth3', 'TaiESM1', 'CanESM5']                                                       # have land values
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['not_in_schiro']:
        models_exclude = ['INM-CM5-0', 'IITM-ESM', 'INM-CM4-8', 'KIOST-ESM', 'MIROC-ES2L', 'EC-Earth3', 'CNRM-CM6-1-HR', 'KACE-1-0-G', 'IPSL-CM6A-LR']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['res_versions']:
        models_exclude = ['NorESM2-LM', 'NorESM2-MM', 'CNRM-CM6-1', 'CNRM-CM6-1-HR'] #, 'GFDL-CM4', 'GFDL-ESM4', 'INM-CM5-0', 'INM-CM4-8', 'CMCC-ESM2', 'CMCC-CM2-SR5']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['threshold_res']:
        models_exclude = ['MIROC-ES2L', 'CanESM5', 'GFDL-CM4', 'NorESM2-LM', 'FGOALS-g3', 'NESM3', 'KIOST-ESM', 'MPI-ESM1-2-LR', 'IITM-ESM', 'IPSL-CM6A-LR', 'INM-CM5-0', 'INM-CM4-8'] # in order from lowest to highest res (currently: 2.5 degree res threshold) (first 4 the ones that really stand out)
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])                            

    if switch_subset['unrealistic_org']: # qualitative for now (based on PWAD and ROME frequency of occurance)
        models_exclude = ['INM-CM5-0', 'FGOALS-g3', 'INM-CM4-8', 'MPI-ESM1-2-LR', 'KIOST-ESM', 'BCC-CSM2-MR', 'NESM3', 'CNRM-ESM2-1', 'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'IPSL-CM6A-LR', 'CanESM5', 'NorESM2-LM', 'GFDL-CM4', 'KACE-1-0-G', 'ACCESS-CM2', 'UKESM1-0-LL', 'MIROC6', 'GFDL-ESM4'] 
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])     

## Unrealistic organization behaviour
# models that stand out in PWAD:
# INM-CM5-0         (second most, in small objects)                     1
# FGOALS-g3         (a little bit, small objects)                       2
# INM-CM4-8         (second most tied, small)                           3
# MPI-ESM1-2-LR     (largest objects)                                   4
# KIOST-ESM         (stand out a little bit with large objects)         5
# BCC-CSM2-MR       (somewhat small objects)                            6
# GFDL-ESM4         (somwhat small objects)                             7
# GFDL-CM4          (somewhat small objects)                    
# CMCC-CM2-SR5      (a little bit, small objects)
# CMCC-ESM2         (a little bit)
# NESM              (somewhat large objects, second largest)            8
# CNRM-ESM2-1       (somewhat small objects, smaller than CMCC)         9
# CNRM-CM5-1        (somewhat small)                                    10
# CNRM-CM6-1-HR     (somewhat small)                                    11
# KACE-1-0-G        (a little bit small)                                12
# IPSL-CM6A-LR      (somewhat small, third smallest)                    13
# CanESM5           (stands out the most in small objects)              14
# UKESM1-0-LL       (stands out slightly in small)


# Models that stand out in ROME:
# INM-CM5-0         (second group, small)
# FGOALS            Second most in small ROME
# INM-CM4-8         (second group)
# MPI-ESM1-2-LR     (largest group)
# KIOST-ESM         (largest group)
# BCC-CSM2-MR       (second smallest group)
# GFDL-ESM4         (second smallest group)
# Nor-ESM2-LM       (largest group)
# GFDL-CM4          (second smallest group)
# NESM3             (largest group)
# CNRM-ESM2-1       (second smallest group)
# CNRM-CM6-1        (second smallest group)
# CNRM-CM6-1-HR     (smallest group)
# KACE-1-0-G        (second smallest group)
# IPSL-CM6A-LR      (second smallest)
# ACCESS-CM2        (edge of second smallest group, closest to obs)
# CanESM5           (smallest group, smallest)
# UKESM-1-0-LL      (second smallest group)

    if switch_subset['org_insensitive']: # based on correlation with large-scale state in current climate conditions
        models_exclude = ['CanESM5', 'INM-CM4-8', 'INM-CM5-0', 'BCC-CSM2-MR', 'MPI-ESM1-2-LR', 'NESM3', 'FGOALS-g3', 'IITM-ESM']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded]) 



# org - large-scale state sensitivity: (lowest corr at top)
# pr_99                 
# CanESM5           
# INM-CM4-8         
# INM-CM5-0
# BCC-CSM2-MR
# ACCESS-ESM1-
# CNEM-CM6-1-HR
# IPSL-CM6A-LR
# MIROC6
# CNRM-ESM2-1
# FGOALS-g3
# KACE-1-0-G
# MIROC-ES2L
# NESM3
# KIOST
# ...
#
# (highest) 
# NorESM2-LM
# ITM-ESM
# TaiESM
# CESM2-WACCM
# NorESM2-MM


# relative humidity
# CanESM5
# INM-CM4-8
# FGOALS-g3
# MPI-ESM1-2-LR
# NESM3
# INM-CM5-0
# IITM-EMS
# KACE-1-0-G
# KIOST-ESM
# BCC-CSM2-MR
# GFDL-ESM4
# MRI-ESM2-0
# IPSL-CM6A-LR
#
# ..
# (highest)
# CMCC-CM2-SR5
# ACCESS-CM2
# NorESM2-MM
# NorESM2-LM
# CMCC-ESM2
# CESM2-WACCM
# TaiESM1



# OLR
# MPI-ESM1-2-LR
# FGOALS-g3
# IITM-ESM
# INM-CM5-)
# NESM3
# KIOST-ESM
# INM-CM4-8
# CanESM5
# BCC-CSM2-MR
# 
# ..
# (highest)
# GFDL-CM4
# CMCC-CM2-SR5
# CMCC-ESM2
# NorESM2-LM
# CESM2-WACCM
# TaiESM1
# Nor-ESM2-MM




    # print(models_excluded)
    models_cmip6 = list(filter(lambda x: x not in models_excluded, models_cmip6)) if switch_subset['exclude']      else models_cmip6
    models_cmip6 = models_excluded                                                if switch_subset['only_include'] else models_cmip6
    return models_cmip6, models_excluded
models_cmip6, models_excluded = exclude_models(models_cmip6, switch_subset)




def data_available(source = '', dataset = '', experiment = '', var = '', switch = {'ocean_mask': False}, resolution = 'regridded'):
    ''' Check if dataset has variable '''
    if [source, experiment] == ['cmip5', 'ssp585'] or [source, experiment] == ['cmip6', 'rcp85']: # only run fitting scenario for cmip version
        return  False
    if not experiment and not source in ['obs', 'test']:                                          # only run obs or other for experiment == ''
        return False
    if experiment and source in ['obs']:                                                          # only run models when experiment ~ '' 
        return False
    
    # No clouds
    if var in ['lcf', 'hcf', 'cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid'] and dataset in ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3', 'INM-CM4-8', 'CNRM-CM6-1-HR', 'GFDL-ESM4']:
        print(f'No {var} data for this dataset')
        return False                                                                   

    # No original grid ocean mask
    for mask_type in [k for k, v in switch.items() if v]:
        if mask_type == 'ocean_mask' and resolution == 'orig'and dataset in ['IITM-ESM', 'BCC-CSM2-MR', 'NESM3', 'UKESM1-0-LL', 'CNRM-ESM2-1']: 
            print(f'No original grid ocean mask for this dataset')
            return False
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
    ''' Determining source of dataset list (for plots) '''
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
    ''' Indicate if there is observations in the dataset list (for plots) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''

def get_ds_highlight(switch_highlight, datasets, switch_subset= {'a':False}, func = exclude_models):
    dataset_highlight = []
    for item in [k for k, v in switch_highlight.items() if v]:
        if  item == 'by_dTas':
            dataset_highlight = datasets[-int(len(datasets)/2):] # Largest increase in dTas
        if  item == 'by_org_hur_corr':
            dataset_highlight = ['MIROC6', 'NorESM2-LM', 'TaiESM1']
        if  item == 'by_obs_sim':
            dataset_highlight = ['MIROC6', 'TaiESM1']
        if  item == 'by_excluded':
            switch_subset['only_include'] = True
            _, dataset_highlight = func(datasets, switch_subset)
            # dataset_highlight = ['NorESM2-LM', 'NorESM2-MM', 'CNRM-CM6-1', 'CNRM-CM6-1-HR']
    return dataset_highlight


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
    'INM-CM4-8':     1.83,      
    'INM-CM5-0':     1.92,      
    'IITM-ESM':      2.37,  
    'NorESM2-MM':    2.49,   
    'NorESM2-LM':    2.56, 
    'MIROC6':        2.6,
    'GFDL-ESM4':     2.65,    
    'MIROC-ES2L':    2.66, 
    'FGOALS-g3':     2.87,  
    'MPI-ESM1-2-LR': 3.02,
    'BCC-CSM2-MR':   3.02,      
    'MRI-ESM2-0':    3.13,  
    'KIOST-ESM':     3.36,
    'CMCC-CM2-SR5':  3.56,   
    'CMCC-ESM2':     3.58,
    'CMCC-ESM2':     3.58,    
    'ACCESS-ESM1-5': 3.88,   
    'GFDL-CM4':      3.89,   
    'EC-Earth3':     4.26,   
    'CNRM-CM6-1-HR': 4.34,  
    'TaiESM1':       4.36,   
    'ACCESS-CM2':    4.66,
    'CESM2-WACCM':   4.68,   
    'IPSL-CM6A-LR':  4.70,   
    'NESM3':         4.72,   
    'KACE-1-0-G':    4.75,
    'CNRM-ESM2-1':   4.79, 
    'CNRM-CM6-1':    4.90,        
    'CESM2':         5.15,      
    'UKESM1-0-LL':   5.36,    
    'CanESM5':       5.64,           

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


# ---------------------------------------------------------------------------------------- Order by ----------------------------------------------------------------------------------------------------- #
switch_order = {'ecs': False}
def order_by(models_cmip6, switch_order, ecs_list):
    if switch_order['ecs']:
        models_cmip6 = sorted(models_cmip6, key=lambda model: ecs_list.get(model, float('inf')))
    return models_cmip6
models_cmip6 = order_by(models_cmip6, switch_order, ecs_list)




datasets = models_cmip5 + models_cmip6 + observations + constructed_fields
institutes = {**institutes_cmip5, **institutes_cmip6}













































