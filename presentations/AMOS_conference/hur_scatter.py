'''
# ---------------------
#  hur_mean scatter
# ---------------------
'''
# models_closeObs = ['CESM2-WACCM', 'EC-Earth3', 'MIROC-ES2L', 'MIROC6', 'NorESM2-MM', 'TaiESM1'] #(basd on pr and pwad RMSE)
# models_closeObs = ['CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #(basd on hur and pwad RMSE)

models_closeObs = ['ACCESS-ESM1-5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #, 'ACCESS-CM2']

# 'MIROC6', 'NESM3', 'IITM-ESM', 'IPSL-CM6A-LR', 'NorESM2-LM', 'KIOST-ESM', 'MIROC-ES2L'



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars                   as mV   
import myFuncs                  as mF     
import myFuncs_plots            as mFp

sys.path.insert(0, f'{os.getcwd()}/util-plot')
import scatter_plot_label       as sPl

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.metric_data     as mD



# ------------------------
#   Calculate metric
# ------------------------
# ------------------------------------------------------------------------------------ hur_mean ----------------------------------------------------------------------------------------------------- # 
def calc_hur_mean(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        ds[dataset] = mD.load_metric_file('hur', 'hur_700hpa_sMean', dataset, experiment_alt).mean(dim='time').compute()
    return ds

def get_hur_mean(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'hur_mean_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded hur_mean from temp_calc folder')
    except Exception as e:
        print(f'Failed to load hur_mean from temp_calc folder: {e}')
        print(f'Creating hur_mean for {experiment}')
        ds = calc_hur_mean(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved hur_mean')
    return ds


# ---------------------------------------------------------------------------------------- ROME ----------------------------------------------------------------------------------------------------- # 
def calc_rome_mean(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        ds[dataset] = mD.load_metric_file('conv_org', 'rome_95thprctile', dataset, experiment_alt).mean(dim='time').compute()
    return ds

def get_rome_mean(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'rome_mean_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded rome_mean from temp_calc folder')
    except Exception as e:
        print(f'Failed to load rome_mean from temp_calc folder: {e}')
        print(f'Creating rome_mean for {experiment}')
        ds = calc_rome_mean(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved rome_mean')
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
        print(f'Failed to load dtas from temp_calc folder: {e}')
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
    # print(f'obs year range: {mV.obs_years[0]} (when creating, remove temp_calc to reset)')
    remove_temp_calc = True
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = True
    mFp.remove_test_plots() if remove_plots else None

    hur_historical = get_hur_mean(experiment = mV.experiments[0])
    hur_warm = get_hur_mean(experiment = mV.experiments[1])
    rome_historical = get_rome_mean(experiment = mV.experiments[0])
    rome_warm = get_rome_mean(experiment = mV.experiments[1])
    ds_dtas = get_dtas_ds()
    # print(rome_historical)

    plot = True                                                       # plot hur vs rome
    if plot:
        x_label = r'ROME [km$^2$ K$^{-1}$]'
        y_label = r'Relative humidity [% K$^{-1}$]'
        ds_x = xr.Dataset()
        for dataset in mV.datasets:
            # if dataset != mV.observations[0]:
            ds_x[dataset] = (rome_warm[dataset] - rome_historical[dataset]) / ds_dtas[dataset]

        ds_y = xr.Dataset()
        for dataset in mV.datasets:
            # if dataset != mV.observations[0]:
            ds_y[dataset] = (hur_warm[dataset] - hur_historical[dataset]) / ds_dtas[dataset]        
        
        fig, ax = sPl.plot_scatter_ds(ds_x, ds_y, variable_list = mV.models_cmip6, fig_title = '', x_label = x_label, y_label = y_label, fig_given = False, fig = '', ax = '', color = 'k', models_highlight = models_closeObs)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'rome_hur_difference.png')
        print('plotted rome_hur difference')

        plot = True                                                 # Add models close to obs
        if plot: 
            fig, ax = sPl.plot_scatter_ds(ds_x, ds_y, variable_list = models_closeObs, fig_title = '', x_label = x_label, y_label = y_label, fig_given = True, fig = fig, ax = ax, color = 'g')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'rome_hur_closeObs_difference.png')
            print('plotted rome_hur closeObs difference')












































