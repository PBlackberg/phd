import xarray as xr
import numpy as np

from vars.myFuncs import *
from vars.mse_vars import *


def mse_anomMean(mse):

    aWeights = np.cos(np.deg2rad(mse.lat))
    mse_tMean = mse.mean(dim=('time'), keep_attrs=True)
    mse_sMean = mse.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

    mse_anom = mse - mse_sMean
    mse_anomalyMean = mse_anom.mean(dim=('lat','lon'), keep_attrs=True)


    mse_anomSqrd = mse_anom**2
    dmse = mse_anomSqrd.isel(time=slice(1,None)).data-mse_anomSqrd.isel(time=slice(0,-1)).data

    dmse = xr.DataArray(
        data=dmse,
        dims=['time', 'lat', 'lon'],
        coords={'time': mse_anomSqrd.time.data[0:-1], 'lat': mse_anomSqrd.lat.data, 'lon': mse_anomSqrd.lon.data},
        attrs={'units':''}
        )

    return mse_anomalyMean





if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'
    period = slice('1970-01-01','1999-12-31')
    member_id='r1i1p1f1'

    haveData = False
    if haveData:
        folder = '/Users/cbla0002/Documents/data/cmip6/' + model
        fileName = model + '_mse_' + experiment_id + '.nc'
        path = folder + '/' + fileName
        ds = xr.open_dataset(path)
        mse = ds.mse
    else:
        mse = get_mse(model, experiment_id, period, member_id, saveit)




    mse_tMean = mse.mean(dim=('time'), keep_attrs=True)





    aWeights = np.cos(np.deg2rad(mse.lat))
    mse_sMean = mse.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)





    mse_anom = mse - mse_sMean
    mse_anomalyMean = mse_anom.mean(dim=('lat','lon'), keep_attrs=True)

    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/' + model
        fileName = model + '_mse_anomMean_' + experiment_id + '.nc'
        dataset = xr.Dataset({'mse_anomMean': mse_anomalyMean})
        myFuncs.save_file(dataset, folder, fileName)




    mse_anomSqrd = mse_anom**2
    dmse = mse_anomSqrd.isel(time=slice(1,None)).data-mse_anomSqrd.isel(time=slice(0,-1)).data

    dmse = xr.DataArray(
        data=dmse,
        dims=['time', 'lat', 'lon'],
        coords={'time': mse_anomSqrd.time.data[0:-1], 'lat': mse_anomSqrd.lat.data, 'lon': mse_anomSqrd.lon.data},
        attrs={'units':''}
        )











