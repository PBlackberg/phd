import intake
#import xarray as xr
import os
import myFuncs


models = [
        'IPSL-CM5A-MR', # 1
        'GFDL-CM3',     # 2
        'GISS-E2-H',    # 3
        'bcc-csm1-1',   # 4
        'CNRM-CM5',     # 5
        'CCSM4',        # 6 # cannot concatanate files for historical run
        'HadGEM2-AO',   # 7
        'BNU-ESM',      # 8
        'EC-EARTH',     # 9
        'FGOALS-g2',    # 10
        'MPI-ESM-MR',   # 11
        'CMCC-CM',      # 12
        'inmcm4',       # 13
        'NorESM1-M',    # 14
        'CanESM2',      # 15 # slicing with .sel does not work, 'contains no datetime objects'
        'MIROC5',       # 16
        'HadGEM2-CC',   # 17
        'MRI-CGCM3',    # 18
        'CESM1-BGC'     # 19
        ]


model = models[0] #'GFDL-CM3'


historical = True
rcp85 = False

if historical:
    experiment = 'historical'
    period=slice('1970-01','1999-12')
    ensemble = 'r1i1p1'

    if model == 'GISS-E2-H':
        ensemble = 'r6i1p1'


if rcp85:
    experiment = 'rcp85'
    period=slice('2070-01','2099-12')
    ensemble = 'r1i1p1'

    if model == 'GISS-E2-H':
        ensemble = 'r2i1p1'


ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                model_id = model, 
                                experiment = 'historical',
                                time_frequency = 'day', 
                                realm = 'atmos', 
                                ensemble = 'r1i1p1', 
                                variable= 'pr').to_dataset_dict()

if not (model == 'CanESM2' and experiment == 'historical'):
    ds_orig =ds_dict[list(ds_dict.keys())[-1]].sel(time=period, lon=slice(0,360),lat=slice(-35,35))
else:
    ds_orig =ds_dict[list(ds_dict.keys())[-1]].isel(time=slice(43800, 43800+10950)).sel(lon=slice(0,360),lat=slice(-35,35))




haveDsOut = True
ds_pr = myFuncs.regrid_conserv(ds_orig, haveDsOut) # path='', model'')



folder = '/g/data/k10/cb4968/data/cmip5/ds'
save = True
if save:
    os.makedirs(folder, exist_ok=True)

    fileName = model + '_precip_' + experiment + '.nc'
    path = folder + '/' + fileName
    if os.path.exists(path):
        os.remove(path)    

    ds_pr.to_netcdf(path)




















