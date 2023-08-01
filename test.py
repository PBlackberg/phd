


import xarray as xr
import matplotlib.pyplot as plt

a = xr.open_dataset('/Users/cbla0002/Documents/data/org/metrics/rome_equal_area/cmip6/BCC-CSM2-MR_rome_equal_area_daily_historical_regridded')
print(a)


plt.figure()
plt.plot(a['rome'])
plt.show()

