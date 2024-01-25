


import xarray as xr

ds = xr.open_dataset('/Users/cbla0002/Desktop/wap_500hpa_itcz_width/TaiESM1_wap_500hpa_itcz_width_monthly_historical_regridded_144x72.nc')
print(ds)
print(ds['wap_500hpa_itcz_width'].data)



