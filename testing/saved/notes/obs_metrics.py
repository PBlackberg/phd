import xarray as xr
import numpy as np
from os.path import expanduser
home = expanduser("~")
import skimage.measure as skm
import matplotlib.pyplot as plt
import timeit


from metrics.funcs.orgFuncs import *
from metrics.funcs.prFuncs import *



obs = 'GPCP'



folder = home + '/Documents/data/obs/GPCP'
fileName = 'GPCP_precip_processed.nc'
path = folder + '/' + fileName
ds = xr.open_dataset(path)
precip = ds.precip


start = timeit.default_timer()
pr_percentiles = calc_pr_percentiles(precip)
stop = timeit.default_timer()
print('it takes {} minutes to calculate pr_percentiles for GPCP observations'.format((stop-start)/60))
conv_threshold = pr_percentiles.pr97.mean(dim='time')



start = timeit.default_timer()
rome = calc_rome(precip, conv_threshold)
stop = timeit.default_timer()
print('it takes {} minutes to calculate rome for GPCP observations'.format((stop-start)/60))




start = timeit.default_timer()
rome_n = calc_rome_n(n, precip, conv_threshold)
stop = timeit.default_timer()
print('it takes {} minutes to calculate rome_n for GPCP observations'.format((stop-start)/60))



start = timeit.default_timer()
numberIndex = calc_numberIndex(precip, conv_threshold)
stop = timeit.default_timer()
print('it takes {} minutes to calculate the numberIndex for GPCP observations'.format((stop-start)/60))



start = timeit.default_timer()
pwad = calc_pwad(precip, conv_threshold)
stop = timeit.default_timer()
print('it takes {} minutes to calculate the numberIndex for GPCP observations'.format((stop-start)/60))








saveit = False            
if saveit:  
    fileName = obs + '_rome.nc'              
    dataset = xr.Dataset(
        data_vars = {'rome':rome, 
                        'rome_n':rome_n},
        attrs = {'description': 'ROME based on all and the {} largest contiguous convective regions in the scene for each day'.format(n),
                    'units':'km^2'}                  
            )

    save_file(dataset, folder, fileName)


saveit = False
if saveit:
    fileName = obs + '_numberIndex.nc'
    dataset = numberIndex

    save_file(dataset, folder, fileName) 


saveit = False
if saveit:
    fileName = obs + '_pwad.nc'
    dataset = pwad

    save_file(dataset, folder, fileName) 






