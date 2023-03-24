import xarray as xr
import numpy as np

from vars.sef_vars import *


def netsef_anomMean(netsef):

    # netsef_tMean = netsef.mean(dim = 'time', keep_attrs=True)

    aWeights = np.cos(np.deg2rad(netsef.lat))
    netsef_sMean = netsef.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)
    netsef_anom = netsef - netsef_sMean
    netsef_anomalyMean = netsef_anom.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

    return netsef_anomalyMean





if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'
    period = slice('1970-01','1999-12')
    member_id='r1i1p1f1'

    haveData = False
    if haveData:
        folder = '/Users/cbla0002/Documents/data/cmip6/' + model
        fileName = model + '_netlw_' + experiment_id + '.nc'
        path = folder + '/' + fileName
        ds = xr.open_dataset(path)
        netlw = ds.netsef
    else:
        netlw = get_netsef(model, experiment_id, period, member_id, saveit)







    netsw_tMean = netsef.mean(dim = 'time', keep_attrs=True)







    aWeights = np.cos(np.deg2rad(netsef.lat))
    netsef_sMean = netsef.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)







    netsef_anom = netsef - netsef_sMean
    netsef_anomalyMean = netsef_anom.mean(dim = 'time', keep_attrs=True)

    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_netsef_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netsef': netlw})
        myFuncs.save_file(dataset, folder, fileName)
























