'''
# ------------------------
#  ITCZ width calculation
# ------------------------
Based on vertical pressure velocity at 500 hpa
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ------------------------
#    Calculate metric
# ------------------------
# -------------------------------------------------------------------------------------- itcz width ----------------------------------------------------------------------------------------------------- #
def itcz_width_sMean(da):
    da = da.mean(dim = ['lon'])
    itcz = xr.where(da < 0, 1, np.nan).compute()
    itcz_lats = itcz * da['lat']
    max_lats = itcz_lats.max(dim='lat', skipna=True)
    min_lats = itcz_lats.min(dim='lat', skipna=True)
    return max_lats - min_lats                                # range of lats

def itcz_width(da):
    alist = da.mean(dim = ['time', 'lon']).compute()
    itcz_lats = alist.where(alist < 0, drop = True)['lat']          # ascending region
    return itcz_lats.max() - itcz_lats.min()                        # range of lats


# ----------------------------------------------------------------------------------- area fraction of descent ----------------------------------------------------------------------------------------------------- #
def get_fraction_descent(da, dims):
    da = da.mean(dim = 'time')
    da = (xr.where(da > 0, 1, 0) * dims.aream).sum()                # area of descending region
    return (da / dims.aream.sum())*100                              # fraction of descending region

# @mF.timing_decorator()
# def get_area(da, dim):
#     ''' Area covered in domain [% of domain]. Used for area of ascent and descent (wap) '''
#     mask = xr.where(da > 0, 1, 0)
#     area = ((mask*dim.aream).sum(dim=('lat','lon')) / np.sum(dim.aream)) * 100
#     return area



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import numpy as np
    import xarray as xr
    import matplotlib.pyplot as plt
    import os
    import sys
    home = os.path.expanduser("~")                                        
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myFuncs_plots as mFp   
    import myFuncs as mF
    import myVars as mV
    
    ds = xr.open_dataset(f'{mV.folder_scratch}/sample_data/wap/cmip6/TaiESM1_wap_monthly_historical_regridded_144x72.nc')
    da = ds['wap'].sel(plev = 500e2)
    # print(ds)
    print(da)

    plot_mean_wap500 = False
    if plot_mean_wap500:
        da_mean = da.mean(dim = 'time')
        fig = mFp.plot_scene(da_mean, cmap = 'RdBu')         #, vmin = 0, vmax = 60) #, cmap = 'RdBu')
        mFp.show_plot(fig, show_type = 'show')          # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)

    plot_mean_wap500_lat = False
    if plot_mean_wap500_lat:
        da_mean = da.mean(dim = 'time')
        da_mean_lat = da_mean.mean(dim = 'lon')
        x = da_mean_lat.values
        y = da_mean_lat['lat'].values
        plt.figure(figsize=(8, 6))
        plt.plot(x, y, 'k')
        plt.xlabel('time-lon mean wap')
        plt.ylabel('lat')
        plt.axvline(x=0, color='k', linestyle='--')
        plt.axhline(y=0, color='k', linestyle='--')
        mFp.show_plot(fig, show_type = 'show')          # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)

    plot_area_descent = False
    if plot_area_descent:
        da_mean = da.mean(dim = 'time')
        da_mean_descent = xr.where(da_mean > 0, 1, np.nan)
        print(da_mean_descent )
        fig = mFp.plot_scene(da_mean_descent)         #, vmin = 0, vmax = 60) #, cmap = 'RdBu')
        mFp.show_plot(fig, show_type = 'show')          # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)

    dims = mF.dims_class(da)
    width_sMean = itcz_width_sMean(da)
    width = itcz_width(da)
    descent_fraction = get_fraction_descent(da, dims)

    print(f'The itcz width per time step: {np.round(width_sMean[0:5].data, 2)} degrees latitude')
    print(f'The itcz width is: {np.round(width.data, 2)} degrees latitude')
    print(f'The fraction of descending motion is: {np.round(descent_fraction.data, 2)} % of the tropical domain')








