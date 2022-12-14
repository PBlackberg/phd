import xarray as xr
import scipy
import numpy as np



def get_hus_snapshot_tMean(hus):
    da = hus.fillna(0)
    hus_vInt = xr.DataArray(
        data=-scipy.integrate.simpson(da, hus.plev.data, axis=1, even='last')/hus.plev.data[0],
        dims=['time','lat', 'lon'],
        coords={'time': hus.time.data, 'lat': hus.lat.data, 'lon': hus.lon.data},
        attrs={'description':'massweighted vertical integral of humidity',
        'units': 'g/kg'}
        )

    return hus_vInt.isel(time=0), hus_vInt.mean(dim=('time'), keep_attrs=True)




def calc_hus_sMean(hus):
    aWeights = np.cos(np.deg2rad(hus.lat))
    da = hus.fillna(0)
    hus_vInt = xr.DataArray(-scipy.integrate.simpson(da, hus.plev.data, axis=1, even='last')/hus.plev.data[0],
        dims=['time','lat', 'lon'],
        coords={'time': hus.time.data, 'lat': hus.lat.data, 'lon': hus.lon.data},
        attrs={'description':'massweighted vertical integral of humidity',
        'units': 'g/kg'}
        )
    return hus_vInt.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)




