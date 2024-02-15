'''
# ---------------------
#   hur_mean map
# ---------------------
'''

# models_closeObs = ['CESM2-WACCM', 'EC-Earth3', 'MIROC-ES2L', 'MIROC6', 'NorESM2-MM', 'TaiESM1'] #(basd on pr and pwad RMSE)
# models_closeObs = ['CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #(basd on hur and pwad RMSE)
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
# ------------------------------------------------------------------------------------ hur_tMean ----------------------------------------------------------------------------------------------------- # 
def calc_hur_tMean(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        ds[dataset] = mD.load_metric_file('hur', 'hur_700hpa_tMean', dataset, experiment_alt).compute()
    return ds

def get_hur_tMean(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'hur_tMean_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded hur_tMean from temp_calc folder')
    except Exception as e:
        print(f'Failed to load pr_tMean from temp_calc folder: {e}')
        print(f'Creating hur_tMean for {experiment}')
        ds = calc_hur_tMean(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved hur_tMean')
    return ds


# --------------------------------------------------------------------------------------- dTas ----------------------------------------------------------------------------------------------------- # 
def calc_dtas():
    ds = xr.Dataset()
    for dataset in mV.models_cmip6:
        tas_historical = mD.load_metric_file('tas', 'tas_sMean', dataset, mV.experiments[0]).mean(dim = 'time')
        tas_warm = mD.load_metric_file('tas', 'tas_sMean', dataset, mV.experiments[1]).mean(dim = 'time')
        dtas = tas_warm - tas_historical
        ds[dataset] = dtas.compute()
    return ds

def get_dtas_ds():
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'dtas.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded dtas from temp_calc folder')
    except Exception as e:
        print(f'Failed to load pwad from temp_calc folder: {e}')
        print(f'Creating ds_dtas')
        ds = calc_dtas()
        ds.to_netcdf(path, mode = 'w')
        print(f'saved dtas')
    return ds



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print(f'Relative humidity plots started')
    print(f'obs year range: {mV.obs_years[0]} (when creating)')
    remove_temp_calc = False
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None

    ds_historical = get_hur_tMean(experiment = mV.experiments[0])
    ds_warm = get_hur_tMean(experiment = mV.experiments[1])
    ds_dtas = get_dtas_ds()
    # print(ds_historical)
    # print(dtas)


    plot = False                                                                   # plot mean (individual) 
    if plot:
        label = 'Relative humidity [%]'
        vmin, vmax = 0, 100
        ds = ds_historical
        for dataset in mV.datasets:
            da_mean = ds[dataset]
            fig, ax = mFp.plot_scene(da_mean, cmap = 'Blues', ax_title = dataset, fig_title = f'hur_tMean', label = label, vmin = vmin, vmax = vmax)
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_hur_tMean')
            print('plotted hur_tMean individual')

    plot = False                                                                    # plot mean difference (all together) 
    if plot:
        label = r'Relative humidity [% K${^-1}$]'
        vmin, vmax = -10, 10 #-25, 25
        ds_diff = xr.Dataset()
        for dataset in ds_historical.data_vars:
            if dataset != mV.observations[0]:
                ds_diff[dataset] = (ds_warm[dataset] - ds_historical[dataset]) / ds_dtas[dataset]
        fig, axes = mP.plot_dsScenes(ds_diff, label = f'Relative humidity [% / K]', title = f'hur_tMean_difference', vmin = vmin, vmax = vmax, cmap = 'RdBu_r', variable_list = ds_diff.data_vars)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference.png')
        print('plotted hur_tMean difference together')

    plot = True                                                                   # plot mean difference (model mean) (all models)
    if plot:
        label = r'Relative humidity [% K${^-1}$]'
        vmin, vmax = -7.5, 7.5 #-15, 15
        ds = ds_historical
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_historical = combined.mean(dim='model')

        ds = ds_warm
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_warm = combined.mean(dim='model')

        ds = ds_dtas
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_dtas = combined.mean(dim='model')

        da_mean_difference = (da_mean_warm - da_mean_historical) / da_mean_dtas

        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean.pdf')
        print('plotted hur tMean difference model_mean')

    plot = True                                                                   # plot mean difference (model mean) (models close to obs)
    if plot:
        vmin, vmax = -7.5, 7.5

        ds = ds_historical[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_historical = combined.mean(dim='model')

        ds = ds_warm[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_warm = combined.mean(dim='model')

        ds = ds_dtas[models_closeObs]
        data_arrays = [ds[var] for var in ds.data_vars if var != mV.observations[0]]
        combined = xr.concat(data_arrays, dim='model')
        da_mean_dtas = combined.mean(dim='model')

        da_mean_difference = (da_mean_warm - da_mean_historical) / da_mean_dtas


        fig, ax = mFp.plot_scene(da_mean_difference, cmap = 'RdBu_r', ax_title = 'CMIP6 (model-mean) (closer to obs)', fig_title = '', vmin = vmin, vmax = vmax, label = label)
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean_closeObs.png')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'hur_tMean_difference_modelMean_closeObs.pdf')
        print('plotted hur tMean difference model_mean with models closer to obs')































