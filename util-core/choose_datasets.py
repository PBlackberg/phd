'''
# ------------------------
#    Choose datasets
# ------------------------
Choose datasets to be used in calculation / plots 
(some models will be filtered out from calculation / plot (with message) if there is no data available)
'''



# ---------------------------------------------------------- Constructed / Random fields for testing calc ----------------------------------------------------------------------------------------------------- #
test_fields = [
    # 'constructed'        # 1
    # 'random'             # 2
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


# --------------------------------------------------------------------------  CMIP6 ----------------------------------------------------------------------------------------------------- #
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
    # 'NorESM2-MM',        # 12            # missing variables on gadi at the moment (check)                          
    # 'MRI-ESM2-0',        # 13                            
    # 'GFDL-CM4',          # 14      
    # 'CMCC-CM2-SR5',      # 15                
    # 'CMCC-ESM2',         # 16                                    
    # 'NESM3',             # 17     
    # 'ACCESS-ESM1-5',     # 18 
    # 'CNRM-ESM2-1',       # 19 
    # 'EC-Earth3',         # 20 
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

experiment_years = [
    ['1970-2000', '2070-2100']
]


# -------------------------------------------------------------------------- DYAMOND ----------------------------------------------------------------------------------------------------- #
models_dyamond = [
    # '',
]


# -------------------------------------------------------------------------- NextGEMS ----------------------------------------------------------------------------------------------------- #
models_nextgems = [
    # 'ICON-ESM_ngc2013',
]
years_range = [
    ['2020-2050']
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
    '1998-2022',          # Full GPCP data
    # '1998-2009',          # High offset in high percentile precipitation from GPCP    (affects organization indices)
    # '2010-2022'           # Low offset in high percentile precipitation from GPCP     (affects organization indices)
    ]



# ------------------------
#    General settings
# ------------------------
datasets = test_fields + models_cmip5 + models_cmip6 + models_nextgems + observations 

experiments = [
    'historical',       # current climate conditions simulation        
    # 'rcp85',            # warm scenario for cmip5
    # 'ssp585',           # warm scenario for cmip6
    # 'obs',              # observations
    ]

resolutions = [
    # 'orig',
    'regridded'
    ]
x_res, y_res = [
    2.5, 2.5
    ]

timescales = [
    'daily',
    # 'monthly',
    ]

conv_percentiles = [    # threshold for precipitation rate considered to be assocaited with convection
    # '90',
    '95',               # default
    # '97'
    ]



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print(f'''
    Chosen datasets:{datasets}
    obs:{observations}, obs_years: {obs_years}
    resolution: {resolutions[0]}
    timescale: {timescales[0]}
    convective percentile {conv_percentiles[0]} (if applicable)
    ''')




