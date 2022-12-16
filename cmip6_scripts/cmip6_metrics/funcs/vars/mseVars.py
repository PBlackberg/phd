import intake
import xarray as xr
import scipy
import matplotlib.pyplot as plt
import myFuncs
import myPlots



def get_mse(model, experiment_id):

    if experiment_id == 'historical':
        period = slice('1970-01','1999-12')
        member_id='r1i1p1f1'


    variable_id = 'ta'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    ta=ds.ta



    variable_id = 'zg'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    zg=ds.zg



    variable_id = 'hus'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict() 
                                    
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hus=ds.hus


    c_p = 1.005
    L_v = 2.256e6
    mse = c_p*ta + zg + L_v*hus


    # mse_vInt = xr.DataArray(
    #     data= -scipy.integrate.simpson(mse.data, mse.plev.data, axis=1, even='last'),
    #     dims= ['time','lat', 'lon'],
    #     coords= {'time': mse.time.data, 'lat': mse.lat.data, 'lon': mse.lon.data}
    #     ,attrs= {'units':''}
    #     )
    # del ds, ta, zg, hus


    mse_year = mse.sel(time = slice('1987-01-01','1999-12-30'))
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_mse_' + experiment_id + '.nc'
        dataset = xr.Dataset({'mse_year': mse_year})
        myFuncs.save_file(dataset, folder, fileName)

    return mse





if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'
    period = slice('1970-01-01','1999-12-31')
    member_id='r1i1p1f1'


    variable_id = 'ta'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    ta=ds.ta
    myPlots.plot_snapshot(ta.isel(time=0).sel(plev=1e5), 'Reds', 'ta', model)
    plt.show()



    variable_id = 'zg'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    zg=ds.zg
    myPlots.plot_snapshot(ta.isel(time=0).sel(plev=1e5), 'Greys', 'ta', model)
    plt.show()



    variable_id = 'hus'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict() 
                                    
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hus=ds.hus
    myPlots.plot_snapshot(ta.isel(time=0).sel(plev=1e5), 'Greens', 'ta', model)
    plt.show()


    c_p = 1.005
    L_v = 2.256e6
    mse = c_p*ta + zg + L_v*hus


    # mse_vInt = xr.DataArray(
    #     data= -scipy.integrate.simpson(mse.data, mse.plev.data, axis=1, even='last'),
    #     dims= ['time','lat', 'lon'],
    #     coords= {'time': mse.time.data, 'lat': mse.lat.data, 'lon': mse.lon.data}
    #     ,attrs= {'units':''}
    #     )
    # del ds, ta, zg, hus


    mse_year = mse.sel(time = slice('1987-01-01','1999-12-30'))
    saveit=False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_mse_' + experiment_id + '.nc'
        dataset = xr.Dataset({'mse_year': mse_year})
        myFuncs.save_file(dataset, folder, fileName)
























