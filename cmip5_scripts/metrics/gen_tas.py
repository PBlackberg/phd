import xarray as xr
import numpy as np

def get_tas_snapshot_tMean(tas):
    return tas.isel(time=0), tas.mean(dim='time', keep_attrs=True)




def calc_tas_annual(tas):
    weights = np.cos(np.deg2rad(tas.lat))
    return tas.resample(time='Y').mean(dim='time', keep_attrs=True).weighted(weights).mean(dim=('lat','lon'))









    