import intake

import xarray as xr
import numpy as np

import matplotlib.pyplot as plt

import functions.myFuncs as myFuncs
import myPlots


def get_netlw(model, experiment_id):

    if experiment_id == 'historical':
        period = slice('1970-01','1999-12')
        member_id='r1i1p1f1'



    variable_id = 'rlut'
    table_id='E3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    rlut_3hr=ds.rlut
    rlut = rlut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    del ds, rlut_3hr


    variable_id = 'rlds'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    rlds_3hr=ds.rlds
    rlds = rlds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    del ds, rlds_3hr 


    variable_id = 'rlus'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800*8+8*28+8, (43800+10950)*8+8*36)).sel(lon=slice(0,360),lat=slice(-30,30))
    rlus_3hr=ds.rlus
    rlus = rlus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    del ds, rlus_3hr


    netlw = rlus - rlds - rlut

    return netlw






if __name__ == '__main__':
    
    model='MPI-ESM1-2-HR'
    experiment_id='historical'

    
    period = slice('1970-01','1999-12')
    member_id='r1i1p1f1'



    variable_id = 'rlut'
    table_id='E3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    rlut_3hr=ds.rlut
    rlut = rlut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    myPlots.plot_snapshot(rlut.isel(time=0), 'Purples', 'rlut', model)
    plt.show()



    variable_id = 'rlds'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    rlds_3hr=ds.rlds
    rlds = rlds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    myPlots.plot_snapshot(rlds.isel(time=0), 'Purples', 'rlds', model)
    plt.show()



    variable_id = 'rlus'
    table_id='3hr'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800*8+8*28+8, (43800+10950)*8+8*36)).sel(lon=slice(0,360),lat=slice(-30,30))
    rlus_3hr=ds.rlus
    rlus = rlus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)

    myPlots.plot_snapshot(rlus.isel(time=0), 'Purples', 'rlus', model)
    plt.show()


    netlw = rlus - rlds - rlut


    saveit = True
    if saveit:
        folder = '/g/data/k10/cb4968/data/cmip6/ds'
        fileName = model + '_netlw_' + experiment_id + '.nc'
        dataset = xr.Dataset({'netlw': netlw})
        myFuncs.save_file(dataset, folder, fileName)



