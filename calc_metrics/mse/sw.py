import xarray as xr
import numpy as np

from vars.sw_vars import *


def netsw_anomMean(netsw):

    #netsw_tMean = netsw.mean(dim = 'time', keep_attrs=True)

    aWeights = np.cos(np.deg2rad(netsw.lat))
    netsw_sMean = netsw.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)
    netsw_anom = netsw - netsw_sMean
    netsw_anomalyMean = netsw_anom.mean(dim = 'time', keep_attrs=True)

    return netsw_anomalyMean





if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'

    haveData = False
    if haveData:
        folder = '/Users/cbla0002/Documents/data/cmip6/ds'
        fileName = model + '_netsw_' + experiment_id + '.nc'
        path = folder + '/' + fileName
        ds = xr.open_dataset(path)
        netsw = ds.netsw
    else:
        netsw = get_netsw(model, experiment_id, period, member_id, saveit)




    netsw_tMean = netsw.mean(dim = 'time', keep_attrs=True)





    aWeights = np.cos(np.deg2rad(netsw.lat))
    netsw_sMean = netsw.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)





    netsw_anom = netsw - netsw_sMean
    netsw_anomalyMean = netsw_anom.mean(dim = 'time', keep_attrs=True)

    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/' + model
        fileName = model + '_netsw_anomMean_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netsw_anomMean': netsw_anomalyMean})
        myFuncs.save_file(dataset, folder, fileName)


