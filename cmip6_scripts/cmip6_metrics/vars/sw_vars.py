import intake
import xarray as xr



model='MPI-ESM1-2-HR'
experiment_id='historical'
period = slice('1970-01-01','1999-12-31')
member_id='r1i1p1f1'


table_id='E3hr'
variable_id = 'rsut'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,365),lat=slice(-30,30))
rsut_3hr=ds.rsut
rsut = rsut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)






table_id='3hr'
variable_id = 'rsus'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,365),lat=slice(-30,30))
rsus_3hr=ds.rsus
rsus = rsus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)






table_id='3hr'
variable_id = 'rsds'
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


folder = '/g/data/k10/cb4968/data/cmip6/' + model
fileName = model + '_netsw_' + experiment_id + '.nc'
path = folder + '/' + fileName
xr.Dataset({'netsw': netsw}).to_netcdf(path, encoding=netsw.encoding.update({'zlib': True, 'complevel': 4}))









