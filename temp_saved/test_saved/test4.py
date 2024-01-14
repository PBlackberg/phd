import xarray as xr
import numpy as np

da            = xr.open_dataset('/g/data/k10/cb4968/data/sample_data/cl/cmip6/IITM-ESM_cl_monthly_historical_regridded.nc')['cl']
p_hybridsigma = xr.open_dataset('/g/data/k10/cb4968/data/sample_data/p_hybridsigma/cmip6/IITM-ESM_p_hybridsigma_monthly_historical_regridded.nc')['p_hybridsigma']
wap500        = xr.open_dataset('/g/data/k10/cb4968/data/sample_data/wap/cmip6/IITM-ESM_wap_monthly_historical_regridded.nc')['wap'].sel(plev = 500e2)

plevs1, plevs2 = [250e2, 0], [1500e2, 600e2]
da = da.where((p_hybridsigma <= plevs2[0]) & (p_hybridsigma >= plevs2[1]), 0).max(dim='lev')
da_d = da.where(wap500>0)

aWeights = np.cos(np.deg2rad(da.lat))
da_d_sMean = da_d.mean(dim = ('lat', 'lon'))






















