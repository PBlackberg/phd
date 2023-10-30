

import xarray as xr
import matplotlib.pyplot as plt


a = xr.open_dataset('/Users/cbla0002/Documents/data/metrics/wap/wap_500hpa_d_area/cmip6/FGOALS-g3_wap_500hpa_d_area_monthly_ssp585_regridded.nc')
# print(a)
da= a['wap_500hpa_d_area']
print(da.mean(dim='time'))







