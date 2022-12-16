import xarray as xr
import numpy as np

from vars.lw_vars import *


def netlw_anomMean(netlw):

    #netlw_tMean = netlw.mean(dim = 'time', keep_attrs=True)

    aWeights = np.cos(np.deg2rad(netlw.lat))
    netlw_sMean = netlw.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)
    netlw_anom = netlw - netlw_sMean
    netlw_anomalyMean = netlw_anom.mean(dim = 'time', keep_attrs=True)

    return netlw_anomalyMean





if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'

    haveData = False
    if haveData:
        folder = '/Users/cbla0002/Documents/data/cmip6/ds'
        fileName = model + '_netlw_' + experiment_id + '.nc'
        path = folder + '/' + fileName
        ds = xr.open_dataset(path)
        netlw = ds.netlw
    else:
        netlw = get_netlw(model, experiment_id, period, member_id, saveit)




    netlw_tMean = netlw.mean(dim = 'time', keep_attrs=True)





    aWeights = np.cos(np.deg2rad(netlw.lat))
    netlw_sMean = netlw.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)





    netlw_anom = netlw - netlw_sMean
    netlw_anomalyMean = netlw_anom.mean(dim = 'time', keep_attrs=True)

    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/' + model
        fileName = model + '_netlw_anomMean_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netlw_anomMean': netlw_anomalyMean})
        myFuncs.save_file(dataset, folder, fileName)


