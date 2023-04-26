import intake
import xarray as xr
import functions.myFuncs as myFuncs
import myPlots



def get_netsef(model, experiment_id):


    if experiment_id == 'historical':
        period = slice('1970-01','1999-12')
        member_id='r1i1p1f1'


    variable_id = 'hfls'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hfls=ds.hfls




    variable_id = 'hfss'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hfss=ds.hfss


    netsef = hfls + hfss


    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_netsef_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netsef': netsef})
        myFuncs.save_file(dataset, folder, fileName)

    return netsef







if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'


    period = slice('1970-01-01','1999-12-31')
    member_id='r1i1p1f1'



    variable_id = 'hfls'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hfls=ds.hfls

    myPlots.plot_snapshot(hfls.isel(time=0), 'viridis')


    variable_id = 'hfss'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hfss=ds.hfss

    myPlots.plot_snapshot(hfss.isel(time=0), 'viridis')


    netsef = hfls + hfss



    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_netsef_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netsef': netsef})
        myFuncs.save_file(dataset, folder, fileName)

















