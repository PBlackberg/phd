import numpy as np
import xarray as xr
import intake
#import xesmf as xe





# Haversine formula (Great circle distance) (takes vectorized input)
def haversine_dist(lat1, lon1, lat2, lon2):

   # radius of earth in km
    R = 6371

    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180)     
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)

    # Haversine formula
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2

    # distance from Haversine function:
    # (1) h = sin(theta/2)^2

    # central angle, theta:
    # (2) theta = d_{great circle} / R
    
    # (1) in (2) and rearrange for d gives
    # d = R * sin^-1(sqrt(h))*2 

    return 2 * R * np.arcsin(np.sqrt(h))






# Connect objects across boundary (objects that touch across lon=0, lon=360 boundary are the same object) (takes array(lat, lon))
def connect_boundary(array):
    s = np.shape(array)
    for row in np.arange(0,s[0]):
        if array[row,0]>0 and array[row,-1]>0:
            array[array==array[row,0]] = min(array[row,0],array[row,-1])
            array[array==array[row,-1]] = min(array[row,0],array[row,-1])







# # conservative interpolation
def regrid_conserv(ds_in, haveDsOut, path='/g/data/k10/cb4968/data/cmip5/FGOALS-g2/FGOALS-g2_ds_regid_historical.nc', modelDsOut='FGOALS-g2'):

    if haveDsOut:
        ds_out = xr.open_dataset(path)
        regrid = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
    
    else:
        ds_dict = intake.cat.nci['esgf'].cmip5.search(
                                        model_id = modelDsOut, 
                                        experiment = 'historical',
                                        time_frequency = 'day', 
                                        realm = 'atmos', 
                                        ensemble = 'r1i1p1', 
                                        variable= 'pr').to_dataset_dict()

        ds_regrid = ds_dict[list(ds_dict.keys())[-1]].sel(time='1970-01-01', lon=slice(0,360),lat=slice(-30,30))
        ds_regrid.to_netcdf(path)
        
    return regrid(ds_in)




