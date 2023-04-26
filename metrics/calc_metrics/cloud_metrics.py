import numpy as np
import xarray as xr
import scipy
import timeit
import os


run_on_gadi = True
if not run_on_gadi:
    home = os.path.expanduser("~") + '/Documents'



import sys
sys.path.insert(0, '{}phd/functions'.format(home))
from myFuncs import *




import constructed_fields as cf





def snapshot(var):
    return var.isel(time=0)

def tMean(var):
    return var.mean(dim='time', keep_attrs=True)

def sMean(var):
    aWeights = np.cos(np.deg2rad(var.lat))
    return var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)


def in_descent(var, dataset, experiment):
    wap500 = get_dsvariable('wap', dataset, experiment, resolution=resolutions[0])['wap'].sel(plev = 5e4)

    if len(var)<1000:
        wap500 = resample_timeMean(wap500, 'monthly')
        wap500 = wap500.assign_coords(time=data.time)
    
    return var.where(wap500>0, np.nan)




































































































