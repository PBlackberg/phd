'''
# ------------------------
#     Regrid xesmf
# ------------------------
This script conservatively horizontally interpolates a dataset using the xesmf package (quicker than raw calculation)
'''



# ------------------------------------------------------------------------------------ Packages --------------------------------------------------------------------------------------------------------- #
# import xesmf as xe # imported in function
import xarray as xr



# ------------------------
#     Regrid xesmf
# ------------------------
# --------------------------------------------------------------------------- Choose model to interpolate to --------------------------------------------------------------------------------------------------------- #
def regrid_conserv_xesmf(ds_in):
    ''' Creates regridder, interpolating to the grid of FGOALS-g2 from cmip5 '''
    import xesmf as xe
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    ds_out = xr.open_dataset('{}/{}'.format(folder, fileName)).sel(lat=slice(-30,30))
    # regridder = xe.Regridder(ds_in, ds_out, 'conservative', periodic=True)
    regridder = xe.Regridder(ds_in.isel(time=0), ds_out, 'conservative', periodic=True)
    return regridder






