
import matplotlib.pyplot as plt
import xarray as xr



def pick_wap_region():
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    da = xr.load_dataset('/Users/cbla0002/Documents/data/cl/sample_data/cmip6/TaiESM1_cl_monthly_historical_regridded.nc')['cl']
    p_hybridsigma = xr.load_dataset('/Users/cbla0002/Documents/data/cl/sample_data/cmip6/TaiESM1_p_hybridsigma_monthly_historical_regridded.nc')['p_hybridsigma']
    da = da.where((p_hybridsigma <= 1500e2) & (p_hybridsigma >= 600e2), 0).max(dim='lev')
    
    wap = xr.load_dataset('/Users/cbla0002/Documents/data/wap/sample_data/cmip6/TaiESM1_wap_monthly_historical_regridded.nc')['wap']
    wap500 = wap.sel(plev = 500e2)
    da = da.where(wap500>0)
    return da


da = pick_wap_region()
da.isel(time=0).plot()
plt.show()


print('finsihed')

