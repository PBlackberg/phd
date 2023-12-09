



import xarray as xr
import matplotlib.pyplot as plt

a = xr.open_dataset('/Users/cbla0002/Documents/data/metrics/h/h_sMean/cmip6/INM-CM5-0_h_sMean_monthly_historical_regridded.nc')
print(a)


plt.plot(a['h_sMean'])
plt.show()





