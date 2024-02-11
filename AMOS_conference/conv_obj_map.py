'''
# ---------------------
#      pr_mean
# ---------------------

'''
models_closeObs = ['ACCESS-ESM1-5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #, 'ACCESS-CM2']


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars               as mV   
import myFuncs              as mF     
import myFuncs_plots        as mFp

sys.path.insert(0, f'{os.getcwd()}/util-plot')
import map_plot             as mP

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.metric_data     as mD



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------ pr_tMean ----------------------------------------------------------------------------------------------------- # 
def calc_o_heatmap(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        ds[dataset] = mD.load_metric_file('conv_org', 'o_heatmap_95thprctile', dataset, experiment_alt).compute()
    return ds

def get_o_heatmap(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'o_heatmap_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded or_tMean from temp_calc folder')
    except Exception as e:  # It's a good practice to catch specific exceptions
        print(f'Failed to load pr_tMean from temp_calc folder: {e}')
        print(f'Creating o_heatmap for {experiment}')
        ds = calc_o_heatmap(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved o_heatmap')
    return ds



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print(f'object heatmap plots started')
    print(f'obs year range: {mV.obs_years[0]} (when creating)')
    remove_temp_calc = False
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None

    ds_historical = get_o_heatmap(mV.experiments[0])*100
    ds_warm = get_o_heatmap(mV.experiments[1])*100
    # print(ds_historical)
    label = 'Frequency of occurence [%]'


# ----------------------------------------------------------------------------------------- plot ----------------------------------------------------------------------------------------------------- # 
    plot = False                                                                   # plot mean (individual) 
    if plot:
        vmin, vmax = 0, 25
        ds = ds_historical
        for dataset in mV.datasets:
            da_mean = ds[dataset]
            fig, ax = mFp.plot_scene(da_mean, cmap = 'Blues', ax_title = dataset, fig_title = f'o_heatmap', label = label, vmin = vmin, vmax = vmax)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_pr_tMean')
            print('plotted o_heatmap individual')

    plot = False                                                                    # plot mean difference (all together) 
    if plot:
        vmin, vmax = -25, 25
        ds_diff = xr.Dataset()
        for dataset in ds_historical.data_vars:
            if dataset != mV.observations[0]:
                ds_diff[dataset] = ds_warm[dataset] - ds_historical[dataset]
        fig, axes = mP.plot_dsScenes(ds_diff, label = label, title = f'o_heatmap_change', vmin = vmin, vmax = vmax, cmap = 'RdBu_r', variable_list = ds_diff.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_difference.png')
        print('plotted o_heatmap difference together')

    plot = True                                                                   # plot mean difference (model mean) (all models)
    if plot:
        vmin, vmax = -17.5, 17.5
        ds = ds_historical
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_historical = combined.mean(dim='model')

        ds = ds_warm
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_warm = combined.mean(dim='model')

        da_mean_difference = da_mean_warm - da_mean_historical

        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_difference_modelMean.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_difference_modelMean.pdf')
        print('plotted tMean difference model_mean')

    plot = True                                                                   # plot mean difference (model mean) (models close to obs)
    if plot:
        vmin, vmax = -17.5, 17.5

        ds = ds_historical[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_historical = combined.mean(dim='model')

        ds = ds_warm[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_warm = combined.mean(dim='model')

        da_mean_difference = da_mean_warm - da_mean_historical


        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean) (closer to obs)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_difference_modelMean_closeObs.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'o_heatmap_difference_modelMean_closeObs.pdf')
        print('plotted tMean difference model_mean with models closer to obs')
































