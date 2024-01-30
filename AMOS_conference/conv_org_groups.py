'''
# ---------------------
#   Conv_org groups
# ---------------------
This script compares organization metrics from models with organization metrics from observations, and ranks closeness to observations.
It also divides models into three categories (close to observations, small object bias, large-object bias)
metrics:
    Probability density function (time-mean)
    Relative frequency of occurance (temporal)
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myFuncs              as mF     



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------------ PWAD --------------------------------------------------------------------------------------------------- #
def r_eff(area):
    ''' Effective radius '''
    return np.sqrt(area/np.pi)

def get_r_gridbox(da):
    ''' Average gridbox size '''
    dims = mF.dims_class(da)
    r_gridbox = r_eff(dims.aream.mean())
    return r_gridbox

def get_x_bins_gridbox(alist_areaMax, bin_width):
    ''' bins with binwidth equal to average gridbox effective radius '''
    x_bins = np.arange(0, r_eff(alist_areaMax.max()) + bin_width, bin_width)
    return x_bins

def get_y_bins_pwad(o_area, pr_o, x_bins):
    ''' Precipitation Weighted Area Distribution (PWAD) [%] (inclusive left)'''
    r_area = r_eff(o_area)
    o_area_w = (o_area * pr_o).sum()
    y_bins = []
    for i in np.arange(0,len(x_bins)-1):
        y_value = (o_area.where((r_area >= x_bins[i]) & (r_area < x_bins[i+1])) * pr_o).sum() / o_area_w
        y_bins.append(y_value.data)
    return xr.DataArray(y_bins)

def get_pwad_all():
    path = '/scratch/w40/cb4968/test/bin_width_pwad.nc'                 # bin_width
    try:
        bin_width = xr.open_dataset(path)['bin_width']
        print('loaded bin_width from temp file')
    except:
        switch = {'test_sample': False, 'ocean_mask': False}
        da, region = vD.get_variable_data(switch, var_name = 'pr', dataset = 'TaiESM1', experiment = 'historical') # just need it for the dimensions 
        bin_width = get_r_gridbox(da)
        ds = xr.Dataset({'bin_width': bin_width})
        ds.to_netcdf(path, mode = 'w')
        print('saved bin_width')

    path = '/scratch/w40/cb4968/test/x_bins_pwad_all_historical.nc'     # x_bins
    try:
        x_bins = xr.open_dataset(path)['x_bins']
        print('loaded x_bins from temp file')
    except:
        print(f'Creating xarray dataset with x_bins from all datasets')
        metric_type_area = 'conv_org'
        metric_name_area = 'o_area_95thprctile'
        alist_areaMax = []
        for dataset, experiment in mF.run_dataset(): 
            if dataset in mV.observations:
                print(f'obs year range: {mV.obs_years[0]}')
            alist_area = mD.load_metric_file(metric_type_area, metric_name_area, dataset, experiment)
            alist_areaMax.append(alist_area.max())
        alist_areaMax = xr.DataArray(alist_areaMax)    
        x_bins = get_x_bins_gridbox(alist_areaMax, bin_width) # list of max area of objects from different datasets
        ds = xr.Dataset(data_vars = {'x_bins': xr.DataArray(x_bins)})
        ds.to_netcdf(path, mode = 'w')
        print('saved x_bins')

    path = '/scratch/w40/cb4968/test/pwad_all_historical.nc'            # dataset with pwad from all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded pwad from temp file')
    except:
        metric_type_area = 'conv_org'
        metric_name_area = 'o_area_95thprctile'
        metric_type_pr = 'pr'
        metric_name_pr = 'pr_o_95thprctile'
        print(f'Creating xarray dataset with pwad from all datasets')
        ds = xr.Dataset()
        for dataset, experiment in mF.run_dataset(): 
            alist_area = mD.load_metric_file(metric_type_area, metric_name_area, dataset, experiment)
            alist_pr = mD.load_metric_file(metric_type_pr, metric_name_pr, dataset, experiment)
            y_bins = get_y_bins_pwad(o_area = alist_area, pr_o = alist_pr, x_bins = x_bins)
            # print(y_bins)
            ds[dataset] = y_bins
        ds.to_netcdf(path, mode = 'w')
        print('saved pwad')
    return ds, x_bins, bin_width


# ----------------------------------------------------------------------------------------- rel freq --------------------------------------------------------------------------------------------------- #
def get_x_bins_frac(alist_max, alist_min = 0, frac = 100):
    ''' Bins as fraction of maximum value '''
    bin_width = (alist_max.max() - alist_min.min()) / frac
    x_bins = np.arange(alist_min.min(), alist_max.max() + bin_width, bin_width)
    return x_bins

def get_y_bins_relFreq(alist, x_bins):
    y_bins = []
    for i in np.arange(0,len(x_bins)-1):
        y_value = xr.where((alist >= x_bins[i]) & (alist < x_bins[i+1]), 1, 0).sum() / len(alist)
        y_bins.append(y_value.data)
    return xr.DataArray(y_bins)

def get_rel_freq_all():
    path = '/scratch/w40/cb4968/test/x_bins.nc'     # x_bins
    try:
        x_bins = xr.open_dataset(path)['x_bins']
        print('loaded x_bins from temp file')
    except:
        print(f'Creating xarray dataset with x_bins from all datasets')
        metric_type = 'conv_org'
        metric_name = 'rome_95thprctile'
        alist_max, alist_min = [], []
        for dataset, experiment in mF.run_dataset(): 
            if dataset in mV.observations:
                print(f'obs year range: {mV.obs_years[0]}')
            alist = mD.load_metric_file(metric_type, metric_name, dataset, experiment)
            alist_max.append(alist.max())
            alist_min.append(alist.min())
        alist_max = xr.DataArray(alist_max)    
        alist_min = xr.DataArray(alist_min)    
        x_bins = get_x_bins_frac(alist_max, alist_min) # list of max area of objects from different datasets
        ds = xr.Dataset(data_vars = {'x_bins': xr.DataArray(x_bins)})
        ds.to_netcdf(path, mode = 'w')
        print('saved x_bins')

    path = '/scratch/w40/cb4968/test/y_bins_all.nc'            # dataset with y_bins from all datasets
    try:
        ds = xr.open_dataset(path)
        print('loaded y_bins from temp file')
    except:
        metric_type = 'conv_org'
        metric_name = 'rome_95thprctile'
        print(f'Creating xarray dataset with pwad from all datasets')
        ds = xr.Dataset()
        for dataset, experiment in mF.run_dataset(): 
            alist = mD.load_metric_file(metric_type, metric_name, dataset, experiment)
            y_bins = get_y_bins_relFreq(alist, x_bins)
            # print(y_bins)
            ds[dataset] = y_bins
        ds.to_netcdf(path, mode = 'w')
        print('saved pwad')
    return ds, x_bins, (x_bins[2]-x_bins[1])



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars                   as mV   

    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars                   as mV   
    import myFuncs                  as mF     
    import myFuncs_plots            as mFp

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import get_data.variable_data   as vD
    import get_data.metric_data     as mD

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import prob_dist_plot           as pP
    import map_plot                 as mP


# ---------------------------------------------------------------------------------- Remove temporary files --------------------------------------------------------------------------------------------------- #
    remove_temp_data, remove_plots, exit_after = False, False, False   # removes the temporary datasets and plots created in sections of this script
    if remove_temp_data:
        mF.remove_test_data()
    if remove_plots:
        mFp.remove_test_plots()
    if exit_after:
        exit()


# ----------------------------------------------------------------------------- tMean: probability density function --------------------------------------------------------------------------------------------------- #
    run_it = False
    if run_it:                                                           # Precipitation Weigthted Area Distribution (PWAD)                                                                    
        plot = False                                                                    # PWAD model and obs
        if plot:
            x_label = 'Object effective radius [km]'
            y_label = 'Fraction of total precipitation []'
            ds, x_bins, bin_width = get_pwad_all()
            # print(ds)
            # exit()

            fig, ax = pP.plot_dsBins(ds, x_bins, bin_width, variable_list = mV.datasets, title = '', x_label = x_label, y_label = y_label)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_model_obs.png')
            print('plotted PWAD model and obs')

        plot = False                                                                  # PWAD change with warming
        if plot:
            ''

# ------------------------------------------------------------------------------ temporal: relative frequency of occurence --------------------------------------------------------------------------------------------------- #
    run_it = False
    if run_it:
            x_label = r'daily ROME [km$^2$]'
            y_label = 'Frequency of occurance'
            ds, x_bins, bin_width = get_rel_freq_all()
            # print(ds)
            # exit()

            fig, ax = pP.plot_dsBins(ds, x_bins, bin_width, variable_list = mV.datasets, title = '', x_label = x_label, y_label = y_label)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'rel_freq_model_obs.png')
            print('plotted rel freq model and obs')





