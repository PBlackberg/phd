
import xarray as xr
da = xr.open_dataset('/Users/cbla0002/Documents/data/pr/sample_data/cmip6/MPI-ESM1-2-LR_pr_daily_historical_regridded.nc')['pr']
da = None


if not da:
    print('executes')









