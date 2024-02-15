'''
# -------------------------------------
#  Mean change (time and spatial mean)
# -------------------------------------
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                   
import myFuncs                  as mF       
import myFuncs_plots            as mFp

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_data   as vD
import get_data.metric_data     as mD

sys.path.insert(0, f'{os.getcwd()}/util-plot')
import scatter_plot_label       as sPl



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------- ROME --------------------------------------------------------------------------------------------------- #
def get_itcz_width():
    print(f'Creating xarray dataset with ITCZ width from all datasets')
    ds = xr.Dataset()
    folder = '/Users/cbla0002/Documents/data/metrics/wap/wap_500hpa_itcz_width/cmip6'
    folder_tas = '/Users/cbla0002/Documents/data/metrics/tas/tas_sMean/cmip6'
    # x = []
    for model in mV.models_cmip6:
        alist_historical = xr.open_dataset(f'{folder}/{model}_wap_500hpa_itcz_width_monthly_historical_regridded_144x72.nc')['wap_500hpa_itcz_width']
        alist_warm = xr.open_dataset(f'{folder}/{model}_wap_500hpa_itcz_width_monthly_ssp585_regridded_144x72.nc')['wap_500hpa_itcz_width']
        metric = alist_warm - alist_historical
        tas_historical = xr.open_dataset(f'{folder_tas}/{model}_tas_sMean_monthly_historical_regridded.nc')['tas_sMean'].mean(dim='time')
        tas_warm = xr.open_dataset(f'{folder_tas}/{model}_tas_sMean_monthly_ssp585_regridded.nc')['tas_sMean'].mean(dim='time')
        tas_change = tas_warm - tas_historical
        ds[model] = metric / tas_change
        # x.append(metric / tas_change)
    return ds

def get_rome_mean():
    print(f'Creating xarray dataset with ROME from all datasets')
    ds = xr.Dataset()
    folder = '/Users/cbla0002/Documents/data/metrics/conv_org/rome_95thprctile/cmip6'
    folder_tas = '/Users/cbla0002/Documents/data/metrics/tas/tas_sMean/cmip6'
    # y = []
    for model in mV.models_cmip6:
        alist_historical = xr.open_dataset(f'{folder}/{model}_rome_95thprctile_daily_historical_regridded_144x72.nc')['rome_95thprctile'].mean(dim='time')
        alist_warm = xr.open_dataset(f'{folder}/{model}_rome_95thprctile_daily_ssp585_regridded_144x72.nc')['rome_95thprctile'].mean(dim='time')
        metric = alist_warm - alist_historical
        tas_historical = xr.open_dataset(f'{folder_tas}/{model}_tas_sMean_monthly_historical_regridded.nc')['tas_sMean'].mean(dim='time')
        tas_warm = xr.open_dataset(f'{folder_tas}/{model}_tas_sMean_monthly_ssp585_regridded.nc')['tas_sMean'].mean(dim='time')
        tas_change = tas_warm - tas_historical
        ds[model] = metric / tas_change
        # y.append(metric / tas_change)
        # np.array(y)
    return ds




if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds_x = get_itcz_width()
    ds_y = get_rome_mean()
    fig, ax = sPl.plot_scatter_ds(ds_x, ds_y, variable_list = mV.models_cmip6, x_label = 'ITCZ width [degrees / K]', y_label = 'ROME [km^2 / K]')
    # plt.show()

    # x = get_itcz_width()
    # y = get_rome_mean()
    # print(x)
    # print(y)

    mFp.show_plot(fig, show_type = 'save_cwd', filename = f'itcz_rome.png')





















