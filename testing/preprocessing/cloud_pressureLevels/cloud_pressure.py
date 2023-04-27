import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cartopy

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import sys
home = os.path.expanduser("~") + '/Documents'
sys.path.insert(0, '{}/phd/functions'.format(home))
from myFuncs import *


resolutions = [
    'original',
    # 'regridded'
    ]

def data_exist(model, experiment, variable):
    data_exist = True
    if variable == 'cl':
        if model == 'CNRM-CM5' or model == 'CCSM4' or model == 'HadGEM2-AO' or model == 'EC-EARTH':
            data_exist = False
        if model == 'CESM1-BGC' and experiment == 'rcp85':
            data_exist = False    
    return data_exist

models_cmip5 = [
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

models_cmip6 = [
    # 'TaiESM1',        # 1
    # 'BCC-CSM2-MR',    # 2
    # 'FGOALS-g3',      # 3
    # 'CNRM-CM6-1',     # 4
    # 'MIROC6',         # 5
    # 'MPI-ESM1-2-HR',  # 6
    # 'NorESM2-MM',     # 7
    # 'GFDL-CM4',       # 8
    # 'CanESM5',        # 9
    # 'CMCC-ESM2',      # 10
    # 'UKESM1-0-LL',    # 11
    # 'MRI-ESM2-0',     # 12
    # 'CESM2',          # 13
    # 'NESM3'           # 14
    ]

datasets = models_cmip5 + models_cmip6

experiments = [
    'historical',
    # 'rcp85'
    ]


for dataset in datasets:
    for experiment in experiments:
        if data_exist(dataset, experiment, 'cl'):
            print(dataset)
            print(experiment) 

            # load data
            ds = get_dsvariable('cl', dataset, experiment, resolution=resolutions[0])
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

            # regrid data 
            data_regrid = regrid_conserv(data)
            p_hybridSigma_regrid = regrid_conserv(p_hybridSigma)

            # find low clouds and high clouds
            cl_low = (data_regrid* xr.where((p_hybridSigma_regrid<=1500e2) & (p_hybridSigma_regrid>=600e2), 1, 0)).max(dim='plev')            
            cl_high = (data_regrid * xr.where((p_hybridSigma_regrid<=250e2) & (p_hybridSigma_regrid>=0), 1, 0)).max(dim='plev')

            # organize into dataset
            ds_cl_low = xr.Dataset({'cl_low':cl_low})
            ds_cl_high = xr.Dataset({'cl_high':cl_high})

            # save
            save_cl = True

            if np.isin(models_cmip5, dataset).any():
                folder_save = '{}/data/cmip5/metrics_cmip5_{}'.format(resolutions[0])
            if np.isin(models_cmip6, dataset).any():
                folder_save = '{}/data/cmip6/metrics_cmip6_{}'.format(resolutions[0])

            if save_cl:
                fileName = dataset + '_cl_low' + experiment + '.nc'              
                save_file(ds_cl_low, folder_save, fileName)

                fileName = dataset + '_cl_high' + experiment + '.nc'              
                save_file(ds_cl_high, folder_save, fileName)










