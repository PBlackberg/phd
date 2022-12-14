import intake
import xarray as xr
import scipy


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



c_p = 1.005
L_v = 2.256e6
mse = c_p*ta + zg + L_v*hus



mse_vInt = xr.DataArray(
    data=-scipy.integrate.simpson(mse.data, mse.plev.data, axis=1, even='last')/mse.plev.data[0], # units of g/kg
    dims=['time','lat', 'lon'],
    coords={'time': mse.time.data, 'lat': mse.lat.data, 'lon': mse.lon.data}
    ,attrs={'units':''}
    )








mse_year = mse.sel(time = slice('1987-01-31','1999-12-31'))
xr.Dataset({'mse_year': mse_year}).to_netcdf('/g/data/k10/cb4968/data/cmip6/' + model + '/' + model + '_mse_' + experiment_id + '.nc', encoding=mse.encoding.update({'zlib': True, 'complevel': 4}))









