''' 
# ------------------------
#      ni_calc
# ------------------------
NI - Number Index
Given a convective regions with roughly fixed area, the number of connected components can describe degree of organization
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ---------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_calc            as vC



# ------------------------------------------------------------------------------------ NI (Number Index) ----------------------------------------------------------------------------------------------------- #
def calc_ni(da):
    ''' Number of object in scene (a rough indicator of clumpiness) '''
    ni = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        da_day = skm.label(da.isel(time=day), background=0,connectivity=2)
        da_day = mF.connect_boundary(da_day)
        labels = np.unique(da_day)[1:]
        o_numberScene = len(labels)
        ni[day] = o_numberScene
    return xr.DataArray(ni, dims = ['time'], coords = {'time': da.time.data})


# -------------------------------------------------------------------------------------- Areafraction ----------------------------------------------------------------------------------------------------- #
def calc_areafraction(da, dim):
    ''' Areafraction convered by convection (with fixed precipitation rate, the convective area will fluctuate from day to day)'''
    areaf = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        areaf_scene = (np.sum(da.isel(time=day) * dim.aream)/np.sum(dim.aream))*100
        areaf[day] = areaf_scene
    return xr.DataArray(areaf, dims = ['time'], coords = {'time': da.time.data})



