'''
# -------------------------
#   Conv_org groups (PWAD)
# -------------------------
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
import myVars              as mV     
import myFuncs             as mF     
import myFuncs_plots       as mFp     

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_data   as vD
import get_data.metric_data     as mD

sys.path.insert(0, f'{os.getcwd()}/util-plot')
import prob_dist_plot           as pP
import bar_plot                 as bP


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

def get_x_bins_gridbox(bin_width):
    ''' bins with binwidth equal to average gridbox effective radius '''
    alist_areaMax = []
    for dataset in mV.datasets:
        for experiment in mV.experiments:
            if experiment == 'historical' and dataset in mV.observations:
                experiment_alt = 'obs'
            elif experiment == 'ssp585' and dataset in mV.observations:
                continue
            else:
                experiment_alt = experiment
            o_area = mD.load_metric_file('conv_org', 'o_area_95thprctile', dataset, experiment_alt)
            alist_areaMax.append(o_area.max())
    alist_areaMax = xr.DataArray(alist_areaMax)    
    x_bins = np.arange(0, r_eff(alist_areaMax.max().data) + bin_width, bin_width)
    return x_bins

def get_y_bins_pwad(o_area, pr_o, x_bins, bin_width):
    ''' Precipitation Weighted Area Distribution (PWAD) [%] (inclusive left)'''
    r_area = r_eff(o_area)
    o_area_w = (o_area * pr_o).sum()
    y_bins = []
    for i in np.arange(0,len(x_bins)-1):
        y_value = (o_area.where((r_area >= x_bins[i]) & (r_area < x_bins[i+1])) * pr_o).sum() / o_area_w
        y_bins.append(y_value.data*100)
    return xr.DataArray(data = y_bins, dims=['bins'], coords={'bins': x_bins[:-1]})

def calc_pwad(experiment = ''):
    da, _ = vD.get_variable_data({'test_sample': False, 'ocean_mask': False}, var_name = 'pr', dataset = 'TaiESM1', experiment = 'historical') # just need it for the dimensions 
    bin_width = get_r_gridbox(da)
    x_bins = get_x_bins_gridbox(bin_width)                              # this create bins long enough to cover any distribution (historical, ssp585, obs)
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        o_area = mD.load_metric_file('conv_org', 'o_area_95thprctile', dataset, experiment_alt)
        pr_o = mD.load_metric_file('pr', 'pr_o_95thprctile', dataset, experiment_alt)
        y_bins = get_y_bins_pwad(o_area, pr_o, x_bins, bin_width)
        ds[dataset] = y_bins
    bins_middle = x_bins[:-1] + 0.5 * bin_width
    ds['bins_middle'] = xr.DataArray(data = bins_middle.data, dims=['bins'], coords={'bins': x_bins[:-1]})
    return ds

def get_pwad_calc_ds(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'pwad_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded pwad from temp_calc folder')
    except Exception as e:  # It's a good practice to catch specific exceptions
        print(f'Failed to load pwad from temp_calc folder: {e}')
        print(f'Creating pwad for {experiment}')
        ds = calc_pwad(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved pwad')
    return ds


# ------------------------------------------------------------------------------------------ RMSE --------------------------------------------------------------------------------------------------- #
def get_d_squared(da_model, da_obs):
    return (da_model - da_obs) ** 2

def get_rmse(da_model, da_obs):
    ''' Root Mean Squared Error '''
    return np.round(np.sqrt(get_d_squared(da_model, da_obs).mean()), 2)    



# ----------------
#      Plot
# ----------------
if __name__ == '__main__':
    print('plotting PWAD prob distribution')
    print(f'with obs range: {mV.obs_years[0]}')
    remove_temp_calc = True
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None


  # ----------------------------------------------------------------------------------- get data ----------------------------------------------------------------------------------------------------- # 
    ds_historical = get_pwad_calc_ds(experiment = mV.experiments[0])
    # ds_warm = get_pwad_calc_ds(experiment = mV.experiments[1])
    # print(ds_historical)
    # exit()


  # ------------------------------------------------------------------------------------- plot ----------------------------------------------------------------------------------------------------- # 
    x_label = 'Object effective radius [km]'
    y_label = 'Fraction of total precipitation [%]'
    
    plot = False                                                               # prob distr (pwad) (individual)
    if plot:
        ds = ds_historical
        fig, ax = pP.plot_dsBins(ds, variable_list = mV.datasets, title = '', x_label = x_label, y_label = y_label)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_historical.png')
        print('plotted historical pwad')


    plot = False                                                               # prob distr (pwad) (all)
    if plot:
        ds = ds_historical
        fig, ax = pP.plot_dsBins(ds, variable_list = mV.datasets, title = '', x_label = x_label, y_label = y_label)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_historical.png')
        print('plotted historical pwad')


    plot = True                                                                        # RMSE
    if plot:
        ds = ds_historical
        ds_rmse = xr.Dataset()
        for dataset in ds.data_vars:
            if dataset == 'bins_middle':
                continue
            if dataset != mV.observations[0]:
                ds_rmse[dataset] = get_rmse(da_model = ds[dataset], da_obs = ds[mV.observations[0]])
        title = '' # f'RMSE_pwad'
        fig, ax = bP.plot_dsBar(ds_rmse, title = title, ylabel = y_label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pwad_barplot.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'RMSE_pwad_barplot.pdf')
        print('plotted RMSE')
        saveit = True
        if saveit:
            folder = f'{mV.folder_scratch}/temp_calc'
            filename = f'pwad_rmse_rank_historical.nc'
            path = f'{folder}/{filename}'
            datasets, alist = [], []
            for dataset in ds_rmse.data_vars:
                datasets.append(dataset)
                alist.append(ds_rmse[dataset])
            sorted_indices = np.argsort(alist)
            sorted_datasets = [datasets[i] for i in sorted_indices]
            ds_rank = xr.Dataset({'model_rank': xr.DataArray(sorted_datasets)})
            ds_rank.to_netcdf(path, mode = 'w')
            print(ds_rank)
            print(ds_rank['model_rank'])
            print('saved pwad RMSE list')


    plot = False                                                               # prob distr (pwad) (all) (shade)
    if plot:
        ds = ds_historical
        fig, ax = pP.plot_dsBins(ds, variable_list = ds.data_vars, title = '', x_label = x_label, y_label = y_label, bins_middle = ds_historical['bins_middle'], bins_given = True, shade = True)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_historical.png')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_historical.pdf')
        print('plotted historical pwad')


    plot = False                                                               # prob distr (pwad) (close to obs) (shade)
    if plot:
        folder = f'{mV.folder_scratch}/temp_calc'
        filename = f'pwad_rmse_rank_historical.nc'
        path = f'{folder}/{filename}'
        models = xr.open_dataset(path)['model_rank'].data[0:14]
        models = np.append(models, 'GPCP')
        ds = ds_historical[models]
        ds_rest = ds_historical.drop_vars(models)
        fig, ax = pP.plot_dsBins(ds_rest, variable_list = ds_rest.data_vars, title = '', x_label = x_label, y_label = y_label, bins_middle = ds_historical['bins_middle'], bins_given = True, shade = False)
        fig, ax = pP.plot_dsBins(ds, variable_list = ds.data_vars, title = '', x_label = x_label, y_label = y_label, bins_middle = ds_historical['bins_middle'], bins_given = True, shade = True, fig = fig, ax = ax, fig_ax_given = True)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_historical.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'pwad_historical_close_to_obs.pdf')
        print('plotted historical pwad')



































