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
    # 'BCC-CSM2-MR',       # 2
    # 'FGOALS-g3',         # 3
    # 'CNRM-CM6-1',        # 4
    # 'MIROC6',            # 5
    # 'MPI-ESM1-2-LR',     # 6
    # 'NorESM2-MM',        # 7
    # 'GFDL-CM4',          # 8
    # 'CanESM5',           # 9
    # 'CMCC-ESM2',         # 10
    # 'UKESM1-0-LL',       # 11
    # 'MRI-ESM2-0',        # 12
    # 'CESM2',             # 19          # -WACCM'
    # 'NESM3',             # 14
    # 'IITM-ESM',          # 15
    # 'EC-Earth3',         # 16
    # 'INM-CM5-0',         # 17
    # 'IPSL-CM6A-LR',      # 18
    # 'KIOST-ESM',         # 19
    ]

observations = [
    # 'GPCP',              # precipitation (from project al33 on gadi)
    # 'ISCCP',              # clouds (weather states) (https://isccp.giss.nasa.gov/wstates/hggws.html)
    # 'CERES',              # radiation (https://ceres-tool.larc.nasa.gov/ord-tool/jsp/EBAFTOA42Selection.jsp#)
    # 'ERA5'
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



# ---------------------------------------------------------------------------------------- Institutes ----------------------------------------------------------------------------------------------------- #

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



