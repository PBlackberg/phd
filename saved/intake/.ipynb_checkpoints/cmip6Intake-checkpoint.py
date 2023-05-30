import xarray as xr
import os
import intake
import xesmf as xe
import scipy


def regrid_conserv_xesmf(ds_in, path_dsOut='/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc', model_dsOut='FGOALS-g2'):

    if path_dsOut:
        ds_out = xr.open_dataset(path_dsOut)
        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
    
    else:
        ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                        model_id = model_dsOut, 
                                        experiment = 'historical',
                                        time_frequency = 'day', 
                                        realm = 'atmos', 
                                        ensemble = 'r1i1p1', 
                                        variable= 'pr').to_dataset_dict()

        ds_out = ds_dict[list(ds_dict.keys())[-1]].sel(time='1970-01-01', lon=slice(0,360),lat=slice(-30,30))

        # ds_regrid= ds_dict[list(ds_dict.keys())[-1]].sel(time='1970-01-01', lon=slice(0,360),lat=slice(-30,30))
        # ds_regrid.to_netcdf(path_saveDsOut)
        # ds_out = xr.open_dataset(path_saveDsOut)

        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
        
    return regrid(ds_in)



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
    ta=ds['ta']



    variable_id = 'zg'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict()

    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    zg=ds['zg']



    variable_id = 'hus'
    table_id='day'
    ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                    source_id=model, 
                                    experiment_id=experiment_id, 
                                    member_id=member_id, 
                                    variable_id=variable_id, 
                                    table_id=table_id).to_dataset_dict() 
                                    
    ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
    hus=ds['hus']


    c_p = 1.005
    L_v = 2.256e6
    g = 9.8
    mse = c_p*ta + zg + L_v*hus

    del ds, ta, zg, hus


    mse = xr.DataArray(
        data= -scipy.integrate.simpson(mse.data, mse.plev.data, axis=1, even='last')/g,
        dims= ['time','lat', 'lon'],
        coords= {'time': mse.time.data, 'lat': mse.lat.data, 'lon': mse.lon.data},
        attrs= {'units':'J m\u00b2'}
        )
    
    ds_mse = xr.DataSet(
        data_vars = {'mse':mse},
        attrs = {'description': 'Vertically integrated moist static energy (simpson\'s method)'}
        )
    
    return ds_mse




def get_lw(model, experiment_id):

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


    lw = rlus - rlds - rlut

    lw.attrs['units'] = 'w m' + chr(0x207B) + '\u00b2'


    ds_lw = xr.DataSet(
        data_vars = {'lw' : lw},
        attrs = {'description': 'net longwave radiation as the difference in radiation terms going into and out off the atmosphere from the surface and TOA: (rlus - rlds - rlut)'}
        )

    return ds_lw




def get_sw(model, experiment_id):

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


    sw = rsds + rsus - rsut

    ds_sw = xr.DataSet(
        data_vars = {'sw' : sw},
        attrs = {'description': 'net longwave radiation as the difference in radiation terms going into and out off the atmosphere from the surface and TOA: (rsus - rsds - rsut)'}
        )

    return ds_sw





def get_sef(model, experiment_id):


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


    sef = hfls + hfss

    ds_sef = xr.DataSet(
        data_vars = {'sef' : sef},
        attrs = {'description': 'sensible and latent heat flux: (hfls + hfss)'}
        )


    return ds_sef



def save_file(dataset, folder, fileName):
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName

    if os.path.exists(path):
        os.remove(path)    
    
    dataset.to_netcdf(path)




if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt


    model = 'MPI-ESM1-2-HR'
    experiment_id = 'historical'


    ds_mse = get_mse(model, experiment_id)
    ds_lw = get_lw(model, experiment_id)
    ds_sw = get_sw(model, experiment_id)
    ds_sef = get_sef(model, experiment_id)



    folder = '/g/data/k10/cb4968/data/cmip5/ds/'
    save_mse = False
    save_lw = False
    save_sw = False
    save_sef = False

    if save_mse:
        fileName = model + '_mse_' + experiment_id + '.nc'
        dataset = ds_mse
        save_file(dataset, folder, fileName)


    if save_lw:
        fileName = model + '_lw_' + experiment_id + '.nc'
        dataset = ds_lw
        save_file(dataset, folder, fileName)


    if save_mse:
        fileName = model + '_sw_' + experiment_id + '.nc'
        dataset = ds_sw
        save_file(dataset, folder, fileName)


    if save_mse:
        fileName = model + '_sef_' + experiment_id + '.nc'
        dataset = ds_sef
        save_file(dataset, folder, fileName)













#         elif experiment == 'ssp585':
#             ds_dict= intake.cat.nci['esgf'].cmip6.search(
#                                             source_id=model, 
#                                             experiment_id=experiment, 
#                                             member_id=ensemble, 
#                                             variable_id=variable, 
#                                             table_id='day').to_dataset_dict()
#             ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=slice('2070-01','2099-12'), lat=slice(-35,35))




# import intake
# ds_dict= intake.cat.nci['esgf']
# ds_dict.cmip6.unique()['table_id']
# model = 'TaiESM1'
# experiment_id = 'ssp585'
# member_id='r1i1p1f1'
# variable_id = 'pr'
# table_id='day'
# # table_id='Amon'

# ds_dict= intake.cat.nci['esgf'].cmip6.search(
#                                 source_id=model, 
#                                 experiment_id=experiment_id, 
#                                 member_id=member_id, 
#                                 variable_id=variable_id, 
#                                 table_id=table_id).to_dataset_dict()

# period=slice('2070-01','2099-12')
# ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lat=slice(-35,35))
# precip=ds['pr']
# precip





