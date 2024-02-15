'''
# ---------------------
# itcz_width scatter
# ---------------------
'''
models_closeObs = ['ACCESS-ESM1-5', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'CMCC-ESM2', 'EC-Earth3', 'MRI-ESM2-0', 'NorESM2-MM', 'TaiESM1'] #, 'ACCESS-CM2'] # from pwad and hur



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
# ------------------------------------------------------------------------------------ itcz_width ----------------------------------------------------------------------------------------------------- # 
def calc_itcz_width(experiment = ''):
    ds = xr.Dataset()
    for dataset in mV.datasets:
        if experiment == 'historical' and dataset in mV.observations:
            experiment_alt = 'obs'
        elif experiment == 'ssp585' and dataset in mV.observations:
            continue
        else:
            experiment_alt = experiment
        ds[dataset] = mD.load_metric_file('wap', 'wap_500hpa_itcz_width', dataset, experiment_alt).compute()
    return ds

def get_itcz_width(experiment = ''):
    folder = f'{mV.folder_scratch}/temp_calc'
    filename = f'itcz_width_{experiment}.nc'
    path = f'{folder}/{filename}'
    try:
        ds = xr.open_dataset(path)
        print(f'loaded itcz_width from temp_calc folder')
    except Exception as e:
        print(f'Failed to load hur_mean from temp_calc folder: {e}')
        print(f'Creating itcz_width ds for {experiment}')
        ds = calc_itcz_width(experiment)
        ds.to_netcdf(path, mode = 'w')
        print(f'saved itcz_width')
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
    print(f'itcz width plots started')
    remove_temp_calc = False
    mF.remove_temp_calc() if remove_temp_calc else None
    remove_plots = False
    mFp.remove_test_plots() if remove_plots else None

    itcz_historical = get_itcz_width(experiment = mV.experiments[0])
    itcz_warm = get_itcz_width(experiment = mV.experiments[1])
    rome_historical = get_rome_mean(experiment = mV.experiments[0])
    rome_warm = get_rome_mean(experiment = mV.experiments[1])
    ds_dtas = get_dtas_ds()

    plot = True                                                       # plot hur vs rome
    if plot:
        x_label = r'ROME [km$^2$ K$^{-1}$]'
        y_label = r'ITCZ width [$\degree$latitude $k^{-1}$]'
        ds_x = xr.Dataset()
        for dataset in mV.datasets:
            ds_x[dataset] = (rome_warm[dataset] - rome_historical[dataset]) / ds_dtas[dataset]

        non_zero_vars = []
        ds_y = xr.Dataset()
        for dataset in mV.datasets:
            value = (itcz_warm[dataset] - itcz_historical[dataset]) / ds_dtas[dataset]     
            if value != 0:
                ds_y[dataset] = value
                non_zero_vars.append(dataset)

        ds_x = ds_x[non_zero_vars]
        fig, ax = sPl.plot_scatter_ds(ds_x, ds_y, variable_list = non_zero_vars, fig_title = '', x_label = x_label, y_label = y_label, fig_given = False, fig = '', ax = '', color = 'k', models_highlight = models_closeObs)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = f'rome_itcz_difference.png')
        print('plotted rome_hur difference')

        plot = True                                                 # Add models close to obs
        if plot: 
            models = np.intersect1d(non_zero_vars, models_closeObs)
            fig, ax = sPl.plot_scatter_ds(ds_x, ds_y, variable_list = models, fig_title = '', x_label = x_label, y_label = y_label, fig_given = True, fig = fig, ax = ax, color = 'g')
            mFp.show_plot(fig, show_type = 'save_cwd', filename = f'rome_itcz_closeObs_difference.pdf')
            print('plotted rome_hur closeObs difference')
















































