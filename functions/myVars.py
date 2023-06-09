import numpy as np

# -------------------------------------------------------------------------------------- For choosing dataset / model ----------------------------------------------------------------------------------------------------- #

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
    'BCC-CSM2-MR',    # 2
    'FGOALS-g3',      # 3
    'CNRM-CM6-1',     # 4
    'MIROC6',         # 5
    'MPI-ESM1-2-HR',  # 6
    'NorESM2-MM',     # 7
    'GFDL-CM4',       # 8
    # 'CanESM5',        # 9
    'CMCC-ESM2',      # 10
    'UKESM1-0-LL',    # 11
    'MRI-ESM2-0',     # 12
    'CESM2',          # 13
    'NESM3'           # 14
    ]


observations = [
    # 'GPCP'
    ]
datasets = models_cmip5 + models_cmip6 + observations


resolutions = [
    # 'orig',
    'regridded'
    ]

experiments = [
    'historical',
    # 'rcp85',
    'ssp585',
    # ''
    ]


def find_source(dataset, models_cmip5, models_cmip6, observations):
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
    source = set()
    for dataset in datasets:
        if dataset in models_cmip5:
            source.add('cmip5')
        elif dataset in models_cmip6:
            source.add('cmip6')
        elif dataset in observations:
            source.add('observations')

    if len(source) == 1:
        return source.pop()
    elif len(source) == 2:
        if 'cmip5' in source and 'cmip6' in source:
            return 'mixed'
        elif 'cmip5' in source and 'observations' in source:
            return 'cmip5'
        elif 'cmip6' in source and 'observations' in source:
            return 'cmip6'
    return

def find_ifWithObs(datasets, observations):
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''


# -------------------------------------------------------------------------------------- Other variables ----------------------------------------------------------------------------------------------------- #

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
    'MPI-ESM1-2-HR':'MPI-M',
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







