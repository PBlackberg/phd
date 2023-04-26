import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy

from os.path import expanduser
home = expanduser("~")

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)



from plotFuncs import *
from functions.myFuncs import *



resolutions = [
    # 'original',
    'regridded'
    ]



def data_exist(model, experiment, variable):
    data_exist = 'yes'

    if variable == 'cl':
        if model == 'CNRM-CM5' or model == 'CCSM4' or model == 'HadGEM2-AO' or model == 'EC-EARTH':
            data_exist = ''
        if model == 'CESM1-BGC' and experiment == 'rcp85':
            data_exist = ''
    
    return data_exist




variable = 'cl'

models = [
    # 'IPSL-CM5A-MR', # 1
    'GFDL-CM3',     # 2
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

observations = [
    ]

datasets = models + observations

experiments = [
    'historical',
    'rcp85'
    ]



for dataset in datasets:
    folder_save = home + '/Documents/data/cmip5/ds_cmip5/' + dataset
    for experiment in experiments:

        if data_exist(dataset, experiment, variable):
            print(dataset)

            ds = get_dsvariable(variable, dataset, experiment, resolution=resolutions[0])
            data = ds['cl']

            # different models have different conversions from height coordinate to pressure coordinate. Need to convert from height coordinate matrix to pressure coordinate matrix
            if dataset == 'IPSL-CM5A-MR' or dataset == 'MPI-ESM-MR' or dataset=='CanESM2':
                p_hybridSigma = ds.ap + ds.b*ds.ps
            elif dataset == 'FGOALS-g2':
                p_hybridSigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
            elif dataset == 'HadGEM2-CC':
                p_hybridSigma = ds.lev+ds.b*ds.orog
            else:
                p_hybridSigma = ds.a*ds.p0 + ds.b*ds.ps

            data_regrid = regrid_conserv(data)
            p_hybridSigma_regrid = regrid_conserv(p_hybridSigma)

            cl_low = (data_regrid* xr.where((p_hybridSigma_regrid<=1.5e5) & (p_hybridSigma_regrid>=6e2), 1, 0)).max(dim='plev')            
            cl_high = (data_regrid * xr.where((p_hybridSigma_regrid<=2.5e4) & (p_hybridSigma_regrid>=0), 1, 0)).max(dim='plev')


            save_clouds = True
            if save_clouds:
                fileName = dataset + '_cloudFraction_test' + experiment + '.nc'              
                ds_cl = xr.Dataset(
                    data_vars = {'cl_low':cl_low, 
                                'cl_high':cl_high},
                    attrs = {'description': 'Cloud fraction calculated as maximum between pressure levels identified from native hybrid sigma-pressure coordinates'}                  
                        )
                save_file(ds_cl, folder_save, fileName)










