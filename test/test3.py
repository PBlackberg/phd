import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

a = xr.open_dataset('/Users/cbla0002/Documents/data/sample_data/stability/IPSL-CM6A-LR_ta_monthly_historical_regridded.nc')
da = a['ta'].isel(time=0)

# da.plot()
# plt.show()


theta =  da * (1000e2 / da.plev)**(287/1005)                  # theta = T (P_0/P)^(R_d/C_p)
print(theta)
theta.sel(plev = 925e2).plot()
plt.show()


plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
da = ((da1 * da1.plev).sum(dim='plev') / da1.plev.sum(dim='plev')) - ((da2 * da2.plev).sum(dim='plev') / da2.plev.sum(dim='plev'))   














