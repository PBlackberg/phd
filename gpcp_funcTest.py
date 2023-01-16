import xarray as xr
import numpy as np
from os.path import expanduser
home = expanduser("~")
import skimage.measure as skm
import matplotlib.pyplot as plt
import timeit



folder = home + '/Documents/data/obs/GPCP'
fileName = 'GPCP_precip_processed.nc'
path = folder + '/' + fileName
ds = xr.open_dataset(path)
precip = ds.precip



from metrics.funcs.aggFuncs import *
from metrics.funcs.prFuncs import *



pr_percentiles = calc_pr_percentiles(precip)
conv_threshold = pr_percentiles.pr97.mean(dim='time')


start = timeit.default_timer()
rome = calc_rome(precip, conv_threshold)
stop = timeit.default_timer()
print('it takes {} minutes to calculate rome for GPCP observations'.format((stop-start)/60))



rome_n = calc_rome_n(n, precip, conv_threshold)
numberIndex = calc_numberIndex(precip, conv_threshold)
pwad = calc_pwad(precip, conv_threshold)



model = 'GPCP'
experiment = 'historical'
saveit = False            
if saveit:  
    fileName = model + '_rome_' + experiment + '.nc'              
    dataset = xr.Dataset(
        data_vars = {'rome':rome, 
                        'rome_n':rome_n},
        attrs = {'description': 'ROME based on all and the {} largest contiguous convective regions in the scene for each day'.format(n),
                    'units':'km^2'}                  
            )

    save_file(dataset, folder, fileName)


saveit = True
if saveit:
    fileName = model + '_numberIndex_' + experiment + '.nc'
    dataset = numberIndex

    save_file(dataset, folder, fileName) 


saveit = True
if saveit:
    fileName = model + '_pwad_' + experiment + '.nc'
    dataset = pwad

    save_file(dataset, folder, fileName) 


