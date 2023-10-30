import xarray as xr
import xesmf as xe

def find_ocean_mask():
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/fx/ocean/fx/r0i0p0/v20161204/sftof'
    filename = 'sftof_fx_FGOALS-g2_historical_r0i0p0.nc'
    ds = xr.open_dataset(f'{folder}/{filename}').sel(lat = slice(-35, 35))
    da = ds['sftof']                                        # binary ocean mask values [0,100]

    import regrid_xesmf as regrid
    regridder = regrid.regrid_conserv_xesmf(ds)             # define regridder based of grid from other model
    da = regridder(da)/100                                  # conservatively interpolate data onto grid from other model
    da = da.where(da == 1)                                  # only keep binary mask (fractional values may not physically represent what the fraction of ocean mean) 
    return da.sel(lat=slice(-30,30))


if __name__ == '__main__':
    da = find_ocean_mask()
    ds = xr.Dataset(data_vars = {'ocean_mask': da}) 
    
    import os
    def save_file(data, folder='', filename='', path = ''):
        if folder and filename:
            path = os.path.join(folder, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        os.remove(path) if os.path.exists(path) else None
        data.to_netcdf(path)

    save_file(ds, folder='/home/565/cb4968/Documents/code/phd/util', filename='ocean_mask.nc')


