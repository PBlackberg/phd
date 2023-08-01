import xarray as xr
import numpy as np

# ------------------------------------------------------------------------------------ regrid function ----------------------------------------------------------------------------------------------------- #

def regrid_conserv(M_in):
    # dimensions of model to regrid to
    folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
    fileName = 'pr_day_FGOALS-g2_historical_r1i1p1_19970101-19971231.nc'
    path1 = folder + '/' + fileName

    folder = '/Users/cbla0002/Documents/data/pr/sample_data/cmip5'
    fileName = 'FGOALS-g2_pr_daily_historical_regridded.nc'
    path2 = folder + '/' + fileName
    
    try:
        M_out = xr.open_dataset(path1)['pr'].sel(lat=slice(-30,30))
    except FileNotFoundError:
        try:
            M_out = xr.open_dataset(path2)['precip'].sel(lat=slice(-30,30))
        except FileNotFoundError:
            print(f"Error: no file at {path1} or {path2}")


    # dimensions
    dlat = M_in.lat.data[1]-M_in.lat.data[0]
    dlon = M_in.lon.data[1]-M_in.lon.data[0]
    latBnds = (M_in.lat.data-(dlat/2), M_in.lat.data+(dlat/2))
    lonBnds = (M_in.lon.data-(dlon/2), M_in.lon.data+(dlon/2))
    lat = np.mean(latBnds, axis=0)
    lon = np.mean(lonBnds, axis=0)
    # area of gridboxes as fraction of earth surface area
    area_wlat = np.cos(np.deg2rad(lat))*dlat*np.pi/(4*180^2)

    dlat_n = M_out.lat.data[1]-M_out.lat.data[0]
    dlon_n = M_out.lon.data[1]-M_out.lon.data[0]
    latBnds_n = (M_out.lat.data-(dlat_n/2), M_out.lat.data+(dlat_n/2))
    lonBnds_n = (M_out.lon.data-(dlon_n/2), M_out.lon.data+(dlon_n/2))
    lat_n = np.mean(latBnds_n, axis=0)
    lon_n = np.mean(lonBnds_n, axis=0)

    # weights
    Wlat = np.zeros([len(lat_n), len(lat)])
    for i in np.arange(0,len(lat_n)):
        latBoxMin_n = latBnds_n[0][i]
        latBoxMax_n = latBnds_n[1][i]

        # gridboxes that are atleast partially overlapping with iteration gridbox
        J = (latBnds[0]<=latBoxMax_n)*(latBnds[1]>= latBoxMin_n)*area_wlat

        # including fractional area component contribution
        I = J*(latBnds[1]-latBoxMin_n)/dlat
        K = J*(latBoxMax_n-latBnds[0])/dlat
        II = np.min([I,J,K], axis=0)

        # weights from individual gridboxes contributing to the new gridbox as fraction of the total combined area contribution
        Wlat[i,:] = II/np.sum(II)

    Wlat = xr.DataArray(
        data = Wlat,
        dims = ['lat_n', 'lat']
        )

    Wlon = np.zeros([len(lon_n), len(lon)])
    for i in np.arange(0,len(lon_n)):
        lonBoxMin_n = lonBnds_n[0][i]
        lonBoxMax_n = lonBnds_n[1][i]

        # gridboxes that are atleast partially overlapping with iteration gridbox
        J = (lonBnds[0]<=lonBoxMax_n)*(lonBnds[1]>= lonBoxMin_n)*1

        # Including fractional area component contribution
        I = J*(lonBnds[1]-lonBoxMin_n)/dlon
        K = J*(lonBoxMax_n-lonBnds[0])/dlon
        L = J*(lonBoxMax_n-lonBnds[0]+360)/dlon
        II = np.min([I,J,K,L], axis=0)

        # weights from individual gridboxes contributing to the new gridbox as fraction of the total combined area contribution
        Wlon[i,:] = II/np.sum(II)

    Wlon = xr.DataArray(
        data = Wlon,
        dims = ['lon_n', 'lon']
        )

    # interpolation
    if 'lev' in M_in.dims or 'plev' in M_in.dims:
        if 'lev' in M_in.dims:
            M_in = M_in.rename({'lev': 'plev'})

        M_n = xr.DataArray(
            data = np.zeros([len(M_in.time.data), len(M_in.plev.data), len(lat_n), len(lon_n)]),
            dims = ['time', 'plev', 'lat_n', 'lon_n'],
            coords = {'time': M_in.time.data, 'plev': M_in.plev.data, 'lat_n': M_out.lat.data, 'lon_n': M_out.lon.data},
            attrs = M_in.attrs
            )

        for day in np.arange(0,len(M_in.time.data)):
            M_Wlat = xr.DataArray(
            data = np.zeros([len(M_in.plev), len(lat_n), len(lon)]),
            dims = ['plev', 'lat_n', 'lon']
            )

            for i in range(0, len(Wlat.lat_n)):
                M_Wlat[:,i,:] = (M_in.isel(time=day) * Wlat[i,:]).sum(dim='lat', skipna=True) / (M_in.isel(time=day).notnull()*1*Wlat[i,:]).sum(dim='lat')
                
            for i in range(0, len(Wlon.lon_n)):
                M_n[day,:,:,i] = (M_Wlat * Wlon[i,:]).sum(dim='lon', skipna=True) / (M_Wlat.notnull()*1*Wlon[i,:]).sum(dim='lon')


    else:
        if 'time' in M_in.dims:
            M_n = xr.DataArray(
                data = np.zeros([len(M_in.time.data), len(lat_n), len(lon_n)]),
                dims = ['time', 'lat_n', 'lon_n'],
                coords = {'time': M_in.time.data, 'lat_n': M_out.lat.data, 'lon_n': M_out.lon.data},
                attrs = M_in.attrs
                )

            for day in np.arange(0,len(M_in.time.data)):
                M_Wlat = xr.DataArray(
                data = np.zeros([len(lat_n), len(lon)]),
                dims = ['lat_n', 'lon']
                )

                for i in range(0, len(Wlat.lat_n)):
                    M_Wlat[i,:] = (M_in.isel(time=day) * Wlat[i,:]).sum(dim='lat', skipna=True) / (M_in.isel(time=day).notnull()*1*Wlat[i,:]).sum(dim='lat')
                    
                for i in range(0, len(Wlon.lon_n)):
                    M_n[day,:,i] = (M_Wlat * Wlon[i,:]).sum(dim='lon', skipna=True) / (M_Wlat.notnull()*1*Wlon[i,:]).sum(dim='lon')

        else:
            M_n = xr.DataArray(
                data = np.zeros([len(lat_n), len(lon_n)]),
                dims = ['lat_n', 'lon_n'],
                coords = {'lat_n': M_out.lat.data, 'lon_n': M_out.lon.data},
                attrs = M_in.attrs
                )

            M_Wlat = xr.DataArray(
            data = np.zeros([len(lat_n), len(lon)]),
            dims = ['lat_n', 'lon']
            )

            for i in range(0, len(Wlat.lat_n)):
                M_Wlat[i,:] = (M_in * Wlat[i,:]).sum(dim='lat', skipna=True) / (M_in.notnull()*1*Wlat[i,:]).sum(dim='lat')
                
            for i in range(0, len(Wlon.lon_n)):
                M_n[:,i] = (M_Wlat * Wlon[i,:]).sum(dim='lon', skipna=True) / (M_Wlat.notnull()*1*Wlon[i,:]).sum(dim='lon')


    M_n = M_n.rename({'lat_n': 'lat', 'lon_n': 'lon'})
    return M_n


# ------------------------------------------------------------------- Get the data from the dataset, regrid, and save ----------------------------------------------------------------------------------------------------- #

def run_regridder(switch, dataset):
    print(f'Running regridder')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    print(f'{dataset} ({source})')

    da = xr.open_dataset('/Users/cbla0002/Documents/data/lw/sample_data/obs/CERES_rlut_monthly_orig.nc')['toa_lw_all_mon']
    print(da.lat[0:3])
    da = regrid_conserv(da)
    print(da.lat[0:3])

    ds = xr.Dataset({f'toa_lw_all_mon' : da})
    folder = '/Users/cbla0002/Documents/data/lw/sample_data/obs'
    filename = 'CERES_rlut_monthly_regridded.nc'
    mV.save_file(ds, folder, filename) if switch['save'] else None





# --------------------------------------------------------------------------------- Choose what to regrid ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    import sys
    import os
    home = os.path.expanduser("~")
    folder_code = f'{home}/Documents/code/phd'
    sys.path.insert(0, f'{folder_code}/functions')
    import myFuncs as mF # imports common operators
    import myVars as mV # imports common variables

    # Chose observation to regrid
    switch = {
        'rlut': True,

        'show': True,
        'save': True
    }

    run_regridder(switch, 
                  dataset = 'CERES', 
                  )









