import os
from pathlib import Path
from getpass import getuser



# ------------------------
#    Choose datasets
# ------------------------
# ------------------------------------------------------ Constructed / Random fields for testing calc ----------------------------------------------------------------------------------------------------- #
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
    # 'MIROC-ES2L',        # 10   # no pr
    # 'NorESM2-LM',        # 11      
    # 'NorESM2-MM',        # 12            # missing variables on gadi at the moment                          
    # 'MRI-ESM2-0',        # 13                            
    # 'GFDL-CM4',          # 14      
    # 'CMCC-CM2-SR5',      # 15                
    # 'CMCC-ESM2',         # 16                                    
    # 'NESM3',             # 17     
    # 'ACCESS-ESM1-5',     # 18   # no pe
    # 'CNRM-ESM2-1',       # 19   # no pe
    # 'EC-Earth3',         # 20 # pe test
    # 'CNRM-CM6-1',        # 21  # no pe
    # 'CNRM-CM6-1-HR',     # 22   # no pe
    # 'KACE-1-0-G',        # 23            
    # 'IPSL-CM6A-LR',      # 24
    # 'ACCESS-CM2',        # 25   # no pe
    # 'TaiESM1',           # 26 # test                           
    # 'CESM2-WACCM',       # 27   
    # 'CanESM5',           # 28  
    # 'UKESM1-0-LL',       # 29  # no pe
    ]


# -------------------------------------------------------------------------- DYAMOND ----------------------------------------------------------------------------------------------------- #
models_dyamond = [
    # '',
]


# -------------------------------------------------------------------------- NextGEMS ----------------------------------------------------------------------------------------------------- #
models_nextgems = [
    # 'ICON-ESM_ngc2013',
]


# -----------------------------------------------------------------------  Observations ----------------------------------------------------------------------------------------------------- #
observations = [
    # 'GPCP',               # Precipitation (and organization index)    - project al33 on nci                                                       (1998-01 2022-12)
    # 'ISCCP',              # Clouds (weather states)                   - https://isccp.giss.nasa.gov/wstates/hggws.html                            (2000-01 2017-12)
    # 'CERES',              # Radiation                                 - https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp#     (2000-03 2023-04)
    # 'ERA5',               # Humidity                                  - project rt52 on gadi                                                      (1998-01 2021-12)
    # 'NOAA'                # surface temperature                       - https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
    ]
obs_years = [
    # '1998-2022',          # Full GPCP data
    # '1998-2009',          # High offset in high percentile precipitation from GPCP    (affects organization indices)
    # '2010-2022'           # Low offset in high percentile precipitation from GPCP     (affects organization indices)
    ]



# ------------------------
#    General settings
# ------------------------
timescales = [
    # 'daily',
    'monthly',
    ]

experiments = [
    'historical',       # current climate conditions simulation        
    # 'rcp85',            # warm scenario for cmip5
    'ssp585',           # warm scenario for cmip6
    # 'obs',              # observations
    ]

resolutions = [
    # 'orig',
    'regridded'
    ]
x_res, y_res = [
    2.5, 2.5
    ]

conv_percentiles = [    # threshold for precipitation rate considered to be assocaited with convection
    # '90',
    '95',               # default
    # '97'
    ]



# --------------------------------
# Folder for saving metrics / data
# --------------------------------
folder_save = (os.path.expanduser("~") + '/Documents/data')
folder_save = ('/g/data/k10/cb4968/data')   if os.path.expanduser("~") == '/home/565/cb4968' else folder_save
folder_save = ('/work/bb1153/b382628/data') if os.path.expanduser("~") == '/home/b/b382628'  else folder_save

folder_scratch = (os.path.expanduser("~") + '/Documents/data/scratch')
folder_scratch = ('/scratch/w40/cb4968')                        if os.path.expanduser("~") == '/home/565/cb4968'    else folder_scratch 
folder_scratch = (Path("/scratch") / getuser()[0] / getuser())  if os.path.expanduser("~") == '/home/b/b382628'     else folder_scratch # /scratch/b/b382628








# --------------------------------------
#  Functions for picking dataset subset
# --------------------------------------
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



datasets = test_fields + models_cmip5 + models_cmip6 + models_nextgems + observations 
institutes = {**institutes_cmip5, **institutes_cmip6}


# ------------------------------------------------------------------------------ test --------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    print(f'Chosen datasets:{datasets}')
    print(f'Chosen subset:{datasets}')



