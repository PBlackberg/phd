import scipy
import xarray as xr

def vertical_integral(var):
    var = var.sel(plev=slice(850e2,0)) # free troposphere (most values at 1000 hPa over land are NaN)
    var.fillna(0) # mountains will be NaN for larger values as well, so setting them to zero
    g = 9.8
    var = xr.DataArray(
        data= -scipy.integrate.simpson(var.data, var.plev.data, axis=1, even='last')/g,
        dims=['time','lat', 'lon'],
        coords={'time': var.time.data, 'lat': var.lat.data, 'lon': var.lon.data}
        )
    return var



