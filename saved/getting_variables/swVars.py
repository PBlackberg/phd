import intake
import xarray as xr
import metrics.get_variables.myFuncs as myFuncs
import myPlots



def get_netsw(model, experiment_id):

    if experiment_id == 'historical':
        period = slice('1970-01','1999-12')
        member_id='r1i1p1f1'


    variable_id = 'rsut'
    table_id='E3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,365),lat=slice(-30,30))
    rsut_3hr=ds.rsut
    rsut = rsut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)



    variable_id = 'rsus'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,365),lat=slice(-30,30))
    rsus_3hr=ds.rsus
    rsus = rsus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)



    variable_id = 'rsds'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    rsds_3hr=ds.rsds
    rsds = rsds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)


    netsw = rsds + rsus - rsut

    return netsw






if __name__ == '__main__':

    model='MPI-ESM1-2-HR'
    experiment_id='historical'
    period = slice('1970-01-01','1999-12-31')
    member_id='r1i1p1f1'


    variable_id = 'rsut'
    table_id='E3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,365),lat=slice(-30,30))
    rsut_3hr=ds.rsut
    rsut = rsut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    myPlots.plot_snapshot(rsut.isel(time=0), 'Oranges')
    



    variable_id = 'rsus'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,365),lat=slice(-30,30))
    rsus_3hr=ds.rsus
    rsus = rsus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    myPlots.plot_snapshot(rsus.isel(time=0), 'Oranges')




    variable_id = 'rsds'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    rsds_3hr=ds.rsds
    rsds = rsds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    myPlots.plot_snapshot(rsds.isel(time=0), 'Oranges')


    netsw = rsds + rsus - rsut



    saveit = False
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_netsw_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netsw': netsw})
        myFuncs.save_file(dataset, folder, fileName)









