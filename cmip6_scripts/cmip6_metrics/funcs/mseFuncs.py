import intake
import xarray as xr

import numpy as np
import scipy

import myFuncs


if __name__ == '__main__':

    # temperature
    model='MPI-ESM1-2-HR'
    experiment_id='historical'
    period = slice('1970-01-01','1999-12-31')
    member_id='r1i1p1f1'



    table_id='day'
    variable_id = 'ta'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))



    ta=ds.ta
    ta_test=ta.isel(time=slice(0, 4)) 



    # geopotential height
    table_id='day'
    variable_id = 'zg'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))



    zg=ds.zg
    zg_test=zg.isel(time=slice(0, 4))




    # humidity
    table_id='day'
    variable_id = 'hus'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict() 
                                    
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))



    hus=ds.hus
    hus_test=hus.isel(time=slice(0, 4))




    # moist static energy
    c_p = 1.005
    L_v = 2.256e6
    mse = c_p*ta + zg + L_v*hus
    mse_test = mse.isel(time=slice(0,4))



    mse_vInt = xr.DataArray(
        data=-scipy.integrate.simpson(mse.data, mse.plev.data, axis=1, even='last')/mse.plev.data[0],
        dims=['time','lat', 'lon'],
        coords={'time': mse.time.data, 'lat': mse.lat.data, 'lon': mse.lon.data}
        ,attrs={'units':''}
        )



    aWeights = np.cos(np.deg2rad(mse.lat))
    mse_tMean = mse_vInt.mean(dim=('time'), keep_attrs=True)
    mse_sMean = mse_vInt.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)



    mseVint_anom = mse_vInt - mse_sMean
    mseVint_var = mseVint_anom**2
    dmse = mseVint_var.isel(time=slice(1,len(mse.time.data))).data - mseVint_var.isel(time=slice(0,len(mse.time.data)-1)).data



    dmse = xr.DataArray(
        data=dmse,
        dims=['time', 'lat', 'lon'],
        coords={'time': mse.time.data[0:len(mse.time.data)-1], 'lat': mse.lat.data, 'lon': mse.lon.data},
        attrs={'units':''}
        )



    del zg_test.encoding['chunksizes']




    # mse test
    save = True
    if save:
        folder = '/g/data/k10/cb4968/cmip6/' + model
        fileName = model + '_mse_test_' + experiment_id + '.nc'
        dataSet = xr.Dataset({
                            'mse_test': mse_test, 
                            'ta_test': ta_test, 
                            'zg_test': zg_test,
                            'hus_test': hus_test})
        myFuncs.save_file(dataSet, folder, fileName)



    # mse tMean
    save = True
    if save:
        folder = '/g/data/k10/cb4968/cmip6/' + model
        fileName = model + '_mse_tMean_' + experiment_id + '.nc'
        dataSet = xr.Dataset({'mse_tMean': mse_tMean})
        myFuncs.save_file(dataSet, folder, fileName)




    # mse variance
    save = True
    if save:
        folder = '/g/data/k10/cb4968/cmip6/' + model
        fileName = model + '_mse_var_' + experiment_id + '.nc'
        dataSet = xr.Dataset({
                            'mse_var': mseVint_var.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True),
                            'mse_anom': mseVint_anom.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True),
                            'mse_sMean': mse_sMean})
        myFuncs.save_file(dataSet, folder, fileName)




    # mse dmse/dt
    save = True
    if save:
        folder = '/g/data/k10/cb4968/cmip6/' + model
        fileName = model + '_mse_dmse_' + experiment_id + '.nc'    
        xr.Dataset({'dmse ': dmse.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)})
        myFuncs.save_file(dataSet, folder, fileName)




