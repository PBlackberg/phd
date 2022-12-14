import intake
import xarray as xr

model='MPI-ESM1-2-HR'
experiment_id='historical'
period = slice('1970-01','1999-12')
member_id='r1i1p1f1'



table_id='E3hr'
variable_id = 'rlut'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
rlut_3hr=ds.rlut
rlut = rlut_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)




table_id='3hr'
variable_id = 'rlds'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
rlds_3hr=ds.rlds
rlds = rlds_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)




table_id='3hr'
variable_id = 'rlus'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800*8+8*28+8, (43800+10950)*8+8*36)).sel(lon=slice(0,360),lat=slice(-30,30))
rlus_3hr=ds.rlus
rlus = rlus_3hr.resample(time='1D').mean(dim='time', keep_attrs=True)




netlw = rlus - rlds - rlut

folder = '/g/data/k10/cb4968/data/cmip6/' + model
fileName = model + '_netlw_' + experiment_id + '.nc'
path = folder + '/' + fileName
xr.Dataset({'netlw': netlw}).to_netcdf(path, encoding=netlw.encoding.update({'zlib': True, 'complevel': 4}))







