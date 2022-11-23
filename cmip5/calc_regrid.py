#import intake
import xarray as xr
import xesmf as xe
import os


def calc_regrid(ds_orig, ds_regrid, haveWeights, model):

    folder = '/g/data/k10/cb4968/cmip5/' + model
    fileName = model + '_regrid_weights' + '.nc'
    path = folder + '/' + fileName

    if haveWeights:
        regrid = xe.Regridder(ds_orig, ds_regrid, 'conservative', periodic=True, weights=path)

    else:
        regrid = xe.Regridder(ds_orig, ds_regrid, 'conservative', periodic=True)
        if os.path.exists(path):
            os.remove(path)    

        regrid.to_netcdf(path)
    
    return regrid(ds_orig)









# Alternatively 
# (redefine bounds as in the documentation)
# lon1, lon2 = zip(*ds_orig.lon_bnds)
# lon1 = list(lon1)
# lon2 = list(lon2)
# lon_b = np.append(lon1[0], lon2)

# lat1, lat2 = zip(*ds_orig.lat_bnds)
# lat1 = list(lat1)
# lat2 = list(lat2)
# lat_b = np.append(lat1[0], lat2)

# grid_in={'lon':ds_orig.lon,'lat':ds_orig.lat, 
#           'lon_b':lon_b,'lat_b':lat_b}


# lon1, lon2 = zip(*ds_regrid.lon_bnds)
# lon1 = list(lon1)
# lon2 = list(lon2)
# lon_b = np.append(lon1[0], lon2)

# lat1, lat2 = zip(*ds_regrid.lat_bnds)
# lat1 = list(lat1)
# lat2 = list(lat2)
# lat_b = np.append(lat1[0], lat2)

# grid_out={'lon':ds_regrid.lon,'lat':ds_regrid.lat, 
#           'lon_b':lon_b,'lat_b':lat_b}

# regrid = xe.Regridder(ds_orig, ds_regrid, 'conservative', periodic=True)





