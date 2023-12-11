


import xarray as xr


a = xr.open_dataset('/g/data/k10/cb4968/data/sample_data/clwvi/cmip6/EC-Earth3_clwvi_monthly_historical_regridded.nc')
print(a['clwvi'])

