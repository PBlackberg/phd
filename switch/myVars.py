import os
from pathlib import Path
from getpass import getuser
import numpy as np



# ------------------------
#    Choose datasets
# ------------------------
# ---------------------------------------------------------------- Constructed / Random fields ----------------------------------------------------------------------------------------------------- #
test_fields = [
    # 'random'             # 1
    # 'constructed'        # 2
    ]


# -------------------------------------------------------------------------  CMIP5 ----------------------------------------------------------------------------------------------------- #
models_cmip5 = [
    # 'IPSL-CM5A-MR',       # 1
    # 'GFDL-CM3',           # 2
    # 'GISS-E2-H',          # 3
    # 'bcc-csm1-1',         # 4
    # 'CNRM-CM5',           # 5
    # 'CCSM4',              # 6
    # 'HadGEM2-AO',         # 7
    # 'BNU-ESM',            # 8
    # 'EC-EARTH',           # 9
    # 'FGOALS-g2',          # 10
    # 'MPI-ESM-MR',         # 11
    # 'CMCC-CM',            # 12
    # 'inmcm4',             # 13
    # 'NorESM1-M',          # 14
    # 'CanESM2',            # 15
    # 'MIROC5',             # 16
    # 'HadGEM2-CC',         # 17
    # 'MRI-CGCM3',          # 18
    # 'CESM1-BGC'           # 19
    ]


# -------------------------------------------------------------------------  CMIP6 ----------------------------------------------------------------------------------------------------- #
models_cmip6 = [         # Models ordered by change in temperature with warming
    # 'INM-CM5-0',         # 1
    # 'IITM-ESM',          # 2
    # 'FGOALS-g3',         # 3    
    # 'INM-CM4-8',         # 4                                
    # 'MIROC6',            # 5                                      
    # 'MPI-ESM1-2-LR',     # 6                         
    # 'KIOST-ESM',         # 7    
    # 'BCC-CSM2-MR',       # 8              
    # 'GFDL-ESM4',         # 9         
    # 'MIROC-ES2L',        # 10   
    # 'NorESM2-LM',        # 11      
    # 'NorESM2-MM',        # 12                                      
    # 'MRI-ESM2-0',        # 13                                  
    # 'GFDL-CM4',          # 14      
    # 'CMCC-CM2-SR5',      # 15                
    # 'CMCC-ESM2',         # 16                                    
    # 'NESM3',             # 17     
    # 'ACCESS-ESM1-5',     # 18   
    # 'CNRM-ESM2-1',       # 19   
    # 'EC-Earth3',         # 20 # pe test
    # 'CNRM-CM6-1',        # 21  
    # 'CNRM-CM6-1-HR',     # 22   
    # 'KACE-1-0-G',        # 23            
    # 'IPSL-CM6A-LR',      # 24
    # 'ACCESS-CM2',        # 25   
    'TaiESM1',           # 26 # test                           
    # 'CESM2-WACCM',       # 27   
    # 'CanESM5',           # 28  
    # 'UKESM1-0-LL',       # 29  
    ]


# -------------------------------------------------------------------------- DYAMOND ----------------------------------------------------------------------------------------------------- #
models_dyamond = [
    # 'winter',
]


# -----------------------------------------------------------------------  Observations ----------------------------------------------------------------------------------------------------- #
observations = [
    # 'GPCP',                # for precipitation and organization index (from project al33 on gadi)
    # 'GPCP_1998-2009',     # GPCP section 1: high offset in high percentile precipitation
    # 'GPCP_2010-2022',     # GPCP section 2: low offset in high percentile precipitation
    # 'ISCCP',              # clouds (weather states) (https://isccp.giss.nasa.gov/wstates/hggws.html) (2000-01 2017-12)
    # 'CERES',              # radiation (https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp#) (2000-03-2023-04)
    # 'ERA5',               # humidity (from project rt52 on gadi) (1998-01 - 2021-12)
    # 'NOAA'                # surface temperature (https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html)
    ]



# ------------------------
#    General settings
# ------------------------
timescales = [
    'daily',
    # 'monthly',
    ]

experiments = [
    'historical',       # current climate conditions simulation        
    # 'rcp85',          # warm scenario for cmip5
    # 'ssp585',         # warm scenario for cmip6
    # '',               # observations
    ]

x_res, y_res = 0.1, 0.1
resolutions = [
    # 'orig',
    'regridded'
    ]

conv_percentiles = [    # threshold for precipitation rate considered to be assocaited with convection
    # '90',
    '95',             # default
    # '97'
    ]



# -------------------------
# Folder for saving metric
# -------------------------
folder_save = [os.path.expanduser("~") + '/Documents/data']
folder_save = ['/g/data/k10/cb4968/data']   if os.path.expanduser("~") == '/home/565/cb4968' else folder_save
folder_save = ['/work/bb1153/b382628/data'] if os.path.expanduser("~") == '/home/b/b382628'  else folder_save


folder_scratch = [os.path.expanduser("~") + '/Documents/data']
folder_scratch = ''  if os.path.expanduser("~") == '/home/565/cb4968'                                          else folder_scratch 
folder_scratch = (Path("/scratch") / getuser()[0] / getuser()) if os.path.expanduser("~") == '/home/b/b382628' else folder_scratch # /scratch/b/b382628

# ------------------------
# Other dataset variables
# ------------------------
# ---------------------------------------------------------------------------------------- Institute list ----------------------------------------------------------------------------------------------------- #
institutes_cmip5 = {'IPSL-CM5A-MR': 'IPSL', 'GFDL-CM3': 'NOAA-GFDL', 'GISS-E2-H': 'NASA-GISS', 'bcc-csm1-1': 'BCC',
                    'CNRM-CM5': 'CNRM-CERFACS', 'CCSM4': 'NCAR', 'HadGEM2-AO': 'NIMR-KMA', 'BNU-ESM': 'BNU',
                    'EC-EARTH': 'ICHEC', 'FGOALS-g2': 'LASG-CESS', 'MPI-ESM-MR': 'MPI-M', 'CMCC-CM': 'CMCC',
                    'inmcm4': 'INM', 'NorESM1-M':'NCC','CanESM2': 'CCCma', 'MIROC5': 'MIROC', 'HadGEM2-CC': 'MOHC',
                    'MRI-CGCM3': 'MRI', 'CESM1-BGC': 'NSF-DOE-NCAR'}

institutes_cmip6 = {'TaiESM1': 'AS-RCEC', 'BCC-CSM2-MR': 'BCC', 'FGOALS-g3': 'CAS', 'FGOALS-f3-L': 'CAS',
                    'CanESM5': 'CCCma', 'CMCC-ESM2': 'CMCC', 'CMCC-CM2-SR5': 'CMCC', 'CNRM-CM6-1': 'CNRM-CERFACS',
                    'CNRM-CM6-1-HR': 'CNRM-CERFACS', 'CNRM-ESM2-1': 'CNRM-CERFACS', 'MIROC6': 'MIROC', 'MIROC-ES2L': 'MIROC',
                    'MPI-ESM1-2-LR': 'MPI-M', 'GISS-E2-1-H': 'NASA-GISS', 'GISS-E2-1-G': 'NASA-GISS', 'NorESM2-MM': 'NCC',
                    'NorESM2-LM': 'NCC', 'GFDL-CM4': 'NOAA-GFDL', 'GFDL-ESM4': 'NOAA-GFDL', 'UKESM1-0-LL': 'MOHC', 
                    'KACE-1-0-G': 'NIMS-KMA', 'MRI-ESM2-0': 'MRI', 'CESM2': 'NCAR', 'CESM2-WACCM': 'NCAR', 'NESM3': 'NUIST',
                    'IITM-ESM': 'CCCR-IITM', 'EC-Earth3': 'EC-Earth-Consortium', 'INM-CM5-0': 'INM', 'INM-CM4-8': 'INM',
                    'IPSL-CM6A-LR': 'IPSL', 'KIOST-ESM': 'KIOST', 'ACCESS-ESM1-5': 'CSIRO', 'ACCESS-CM2': 'CSIRO-ARCCSS'}
institutes = {**institutes_cmip5, **institutes_cmip6}


# ---------------------------------------------------------------------------------------- ECS ----------------------------------------------------------------------------------------------------- #
# ECS taken from supplementary information from:
# https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019GL085782#pane-pcw-figures
# https://www.nature.com/articles/d41586-022-01192-2 # most taken from here (used the column which had the same value for models existing in both sets)

ecs_cmip5 = {'IPSL-CM5A-MR': 0, 'GFDL-CM3': 0, 'GISS-E2-H': 0, 'bcc-csm1-1': 0, 'CNRM-CM5': 0, 'CCSM4': 0, 'HadGEM2-AO': 0,        
            'BNU-ESM': 0, 'EC-EARTH': 0, 'FGOALS-g2': 0, 'MPI-ESM-MR': 0, 'CMCC-CM': 0, 'inmcm4': 0, 'NorESM1-M': 0, 'CanESM2': 0,           
            'MIROC5':  0, 'HadGEM2-CC': 0, 'MRI-CGCM3': 0, 'CESM1-BGC': 0}

ecs_cmip6 = {'INM-CM4-8': 1.83, 'INM-CM5-0': 1.92, 'IITM-ESM': 2.37, 'NorESM2-MM': 2.49, 'NorESM2-LM': 2.56, 'MIROC6': 2.6,
            'GFDL-ESM4': 2.65, 'MIROC-ES2L': 2.66, 'FGOALS-g3': 2.87, 'MPI-ESM1-2-LR': 3.02,'BCC-CSM2-MR': 3.02, 'MRI-ESM2-0': 3.13,  
            'KIOST-ESM': 3.36, 'CMCC-CM2-SR5': 3.56, 'CMCC-ESM2': 3.58, 'CMCC-ESM2': 3.58, 'ACCESS-ESM1-5': 3.88, 'GFDL-CM4': 3.89, 
            'EC-Earth3': 4.26, 'CNRM-CM6-1-HR': 4.34, 'TaiESM1': 4.36, 'ACCESS-CM2': 4.66, 'CESM2-WACCM': 4.68, 'IPSL-CM6A-LR': 4.70,   
            'NESM3': 4.72, 'KACE-1-0-G': 4.75, 'CNRM-ESM2-1': 4.79, 'CNRM-CM6-1': 4.90, 'CESM2': 5.15, 'UKESM1-0-LL': 5.36, 'CanESM5': 5.64}
ecs_list = {**ecs_cmip5, **ecs_cmip6}



# --------------------------------
#  Functions for picking subset
# --------------------------------
# --------------------------------------------------------------------- Exclude models based on conditions ----------------------------------------------------------------------------------------------------- #
def order_datasets_by(models_cmip6, switch_order, ecs_list):
    if switch_order['ecs']:
        models_cmip6 = sorted(models_cmip6, key=lambda model: ecs_list.get(model, float('inf')))
    return models_cmip6

def exclude_cmip6_models(models_cmip6, switch_subset):
    models_excluded = []
    if switch_subset['similar_version']:
        models_exclude = ['GFDL-ESM4', 'CMCC-CM2-SR5', 'CNRM-ESM2-1']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['high_res_version']:
        models_exclude = ['NorESM2-MM', 'CNRM-CM6-1-HR']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['no_clouds']:
        models_exclude = ['INM-CM5-0', 'KIOST-ESM', 'EC-Earth3',  'INM-CM4-8', 'CNRM-CM6-1-HR', 
                          'GFDL-ESM4']                                                              # 'GFDL-ESM4' ps missing from pressure calc, calculation of cloud variable was 
                                                                                                    # just different in UK model # 'CNRM-CM6-1-HR' no future scenario
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['no_stability']:
        models_exclude = ['INM-CM4-8', 'GFDL-CM4', 'CESM2-WACCM', 'KACE-1-0-G']                     # just something with KACE air temperature in future scenario
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['not_in_schiro']:
        models_exclude = ['INM-CM5-0', 'IITM-ESM', 'INM-CM4-8', 'KIOST-ESM', 
                          'MIROC-ES2L', 'EC-Earth3', 'CNRM-CM6-1-HR', 'KACE-1-0-G', 
                          'IPSL-CM6A-LR']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['res_versions']:
        models_exclude = ['NorESM2-LM', 'NorESM2-MM', 'CNRM-CM6-1', 'CNRM-CM6-1-HR']                # 'GFDL-CM4', 'GFDL-ESM4', 'INM-CM5-0', 'INM-CM4-8', 'CMCC-ESM2', 'CMCC-CM2-SR5']
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])

    if switch_subset['threshold_res']:
        models_exclude = ['MIROC-ES2L', 'CanESM5', 'GFDL-CM4', 'NorESM2-LM', 
                          'FGOALS-g3', 'NESM3', 'KIOST-ESM', 'MPI-ESM1-2-LR', 
                          'IITM-ESM', 'IPSL-CM6A-LR', 'INM-CM5-0', 'INM-CM4-8']                     # in order from lowest to highest res (currently: 2.5 degree threshold) (first 4 the ones that really stand out)
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])                            

    if switch_subset['unrealistic_org']: 
        models_exclude = ['INM-CM5-0', 'FGOALS-g3', 'INM-CM4-8', 'MPI-ESM1-2-LR', 
                          'KIOST-ESM', 'BCC-CSM2-MR', 'NESM3', 'CNRM-ESM2-1', 
                          'CNRM-CM6-1', 'CNRM-CM6-1-HR', 'IPSL-CM6A-LR', 'CanESM5', 
                          'NorESM2-LM', 'GFDL-CM4', 'KACE-1-0-G', 'ACCESS-CM2', 
                          'UKESM1-0-LL', 'MIROC6', 'GFDL-ESM4']                                     # qualitative for now (based on PWAD and ROME frequency of occurance)
        models_excluded.extend([m for m in models_exclude if m not in models_excluded])     

    if switch_subset['org_insensitive']:
        models_exclude = ['CanESM5', 'INM-CM4-8', 'INM-CM5-0', 'BCC-CSM2-MR', 
                          'MPI-ESM1-2-LR', 'NESM3', 'FGOALS-g3', 'IITM-ESM']                        # based on correlation with large-scale state in current climate conditions
        models_excluded.extend([m for m in models_exclude if m not in models_excluded]) 

    if switch_subset['org_insensitive']:
        models_exclude = []                                                                         # pick models to exclude
        models_excluded.extend([m for m in models_exclude if m not in models_excluded]) 

    models_cmip6 = list(filter(lambda x: x not in models_excluded, models_cmip6)) if switch_subset['exclude']      else models_cmip6
    models_cmip6 = models_excluded                                                if switch_subset['only_include'] else models_cmip6
    return models_cmip6, models_excluded

def highlight_models(switch_highlight, datasets, switch_subset= {'threshold_res':False}, func = exclude_cmip6_models):
    ''' Picks models to highlight in plots (called from plot scripts) '''
    dataset_highlight = []
    for item in [k for k, v in switch_highlight.items() if v]:
        if  item == 'dTas':
            dataset_highlight = datasets[-int(len(datasets)/2):]    # Bottom and top half in change in warming
        if  item == 'subset_switch':
            switch_subset['only_include'] = True            
            _, dataset_highlight = func(datasets, switch_subset)    # If exclude and only_include are false in subset_switch, the subset models can be highlighted
        if  item == 'custom':
            dataset_highlight = []                                  # pick models
    return dataset_highlight



# ------------------------
#  Create dataset list
# ------------------------
# ------------------------------------------------------------------------ choose subset ----------------------------------------------------------------------------------------------------- #
switch_subset = {
    'exclude':         True,  'only_include':      False,                            
    'similar_version': False, 'high_res_version':   False,
    'no_clouds':       False,
    'no_stability':    False,
    'not_in_schiro':   False, 
    'res_versions':    False,
    'threshold_res':   False,                                   # Threshold: 2.5 degrees dlat x dlon
    'unrealistic_org': False,                                   # Threshold: qualitative for now
    'org_insensitive': False,                                   # Threshold: qualitative for now
    'custom':          False
    }


# ------------------------------------------------------------------------- pick out subset --------------------------------------------------------------------------------------------------- #
models_cmip6, models_excluded = exclude_cmip6_models(models_cmip6, switch_subset)
models_cmip6 = order_datasets_by(models_cmip6, switch_order = {'ecs': False}, ecs_list = ecs_list)   # dataets are originally ordered by change in surface temperature. Here you can order them by ECS
datasets = test_fields + models_cmip5 + models_cmip6 + observations



datasets = models_cmip5 + models_cmip6 + observations + constructed_fields + models_dyamondW + models_dyamondS + models_nextgems
institutes = {**institutes_cmip5, **institutes_cmip6}





