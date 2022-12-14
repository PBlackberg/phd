import intake
import xarray as xr


model='MPI-ESM1-2-HR'
experiment_id='historical'
period = slice('1970-01-01','1999-12-31')
member_id='r1i1p1f1'




table_id='day'
variable_id = 'hfls'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
hfls=ds.hfls





table_id='day'
variable_id = 'hfss'
ds_dict= intake.cat.nci['esgf'].cmip6.search(
                                source_id=model, 
                                experiment_id=experiment_id, 
                                member_id=member_id, 
                                variable_id=variable_id, 
                                table_id=table_id).to_dataset_dict()

ds = ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-30,30))
hfss=ds.hfss




netsef = hfls + hfss


del netsef.encoding['chunksizes']

folder = '/g/data/k10/cb4968/data/cmip6/' + model
fileName = model + '_netsef_' + experiment_id + '.nc'
path = folder + '/' + fileName
xr.Dataset({'netsef': netsef}).to_netcdf(path, encoding=netsef.encoding.update({'zlib': True, 'complevel': 4}))




















