
import xarray as xr
import matplotlib.pyplot as plt

da = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome/obs/GPCP_rome_95thPrctile_daily__regridded.nc')['rome']
print(da.time[1].data)


da = xr.open_dataset('/Users/cbla0002/Documents/data/hur/metrics/hur_sMean/obs/ERA5_hur_sMean_monthly__regridded.nc')['hur_sMean']
print(da.time[1].data)


da = xr.open_dataset('/Users/cbla0002/Documents/data/rad/metrics/rlut_sMean/obs/CERES_rlut_sMean_monthly__regridded.nc')['rlut_sMean']
print(da.time[-1].data)



