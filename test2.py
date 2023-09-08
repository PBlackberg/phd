

import matplotlib.pyplot as plt
import xarray as xr
da = xr.open_dataset('/g/data/k10/cb4968/data/wap/metrics/wap_area_a_snapshot/cmip6/IITM-ESM_wap_area_a_snapshot_monthly_historical_regridded.nc')['wap_area_a_snapshot']
da.plot()







