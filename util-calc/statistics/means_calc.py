'''
# --------------
#  Means calc
# --------------
Calculates the mean of different dimensions
'''


# ----------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import re


# -------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #



# -------------
#  Means calc
# -------------
# -------------------------------------------------------------------------------- different means --------------------------------------------------------------------------------------------------- #
def get_tMean(da):
    ''' time-mean '''
    return da.mean(dim='time', keep_attrs=True)

def get_sMean(da):
    ''' spatial mean (area-weighted) '''
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(dim=('lat','lon'), keep_attrs=True)

def get_zMean(da):
    ''' zonal-time-mean (changes with latitude) '''
    return da.mean(dim='lon', keep_attrs=True)

def get_vMean(da, plevs0 = 850e2, plevs1 = 0):         
    ''' vertical mean:
        free troposphere (as most values at 1000 hPa and 925 hPa over land are NaN)
        Where there are no values associated pressure levels are excluded from the weights '''
    da = da.sel(plev = slice(plevs0, plevs1))
    w = ~np.isnan(da) * da['plev']                      
    da = (da * w).sum(dim='plev') / w.sum(dim='plev') 
    return da


# -------------------------------------------------------------------------------- Pick vertical region --------------------------------------------------------------------------------------------------- #
def get_vert_reg(switch, da):
    ''' Choose how to slice data for visualization '''
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        number = re.findall(r'\d+', met_type)
        if number and 'hpa' in met_type: # if there is a number and hpa in string
            level = int(re.findall(r'\d+', met_type)[0]) * 10**2
            da, region = [da.sel(plev = level), f'_{number[0]}hpa']
        if met_type == 'vMean':
            da, region = [get_vMean(da), 'vMean']
    return da, region



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import os
    import sys
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f'running {script_name}')

    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data     as mD
    import dimensions_data  as dD
    import variable_calc    as vC

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import matplotlib.pyplot    as plt
    import get_plot.show_plots  as sP
    import get_plot.map_plot    as mP
    import get_plot.line_plot   as lP


    switch_var = {
        'h':    True   # mse
        }
    
    switch = {
        'test_sample':          False, 
        'from_scratch_calc':    True,  're_process_calc':  False,                                              # if both are true the scratch file is replaced with the reprocessed version (only matters for calculated variables / masked variables)
        'from_scratch':         True,  're_process':       False,                                               # same as above, but for base variables
        }

    switch_test = {
        'remove_script_plots':          False,       # remove plots generated from this script
        'tMean':                        True,      # from 2D (no plev) - result dim:   (lat, lon)      plots:  (map plot)
        'zMean':                        False,      # from 2D (no plev) -               (lat, time)             (line plots for each lat)
        'ztMean':                       True,       # from 3D           -               (plev, lat)             (heat map)
        'sMean':                        False,      # from 3D           -               (plev, time)            (line plots for each plev) 
        }
    sP.remove_folder_plots(folder = f'{os.getcwd()}/zome_plots/{script_name}')  if switch_test['remove_script_plots']   else None   

    experiment = cD.experiments[0]
    for var_name in [k for k, v in switch_var.items() if v]:
        print(f'var_name: {var_name}')
        ds_tMean = xr.Dataset()
        ds_ztMean = xr.Dataset()
        ds_sMean = xr.Dataset()
        ds_zMean = xr.Dataset()
        for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
            print(f'\t dataset: {dataset}')
            # ----------------------------------------------------------------------------------- Get data -------------------------------------------------------------------------------------------------- #
            da, _ = vC.get_variable(switch_var = {var_name: True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            da.load()
            da_vert, region = get_vert_reg(switch = {'vMean': False, '700hpa': True, '500hpa': False, '250hpa': False}, da = da) if 'plev' in da.dims else [da, '']
            # print(region)
            # exit()
            # print(da)
            # exit()

            # ----------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #    
            if switch_test['tMean']:
                ds_tMean[dataset] = get_tMean(da_vert)          # dims: (lat, lon)

            if switch_test['zMean']:
                ds_zMean[dataset] = get_zMean(da_vert)          # dims: (time, lat)

            if switch_test['ztMean']:
                ds_ztMean[dataset] = get_tMean(get_zMean(da))   # dims: (plev, lat)

            if switch_test['sMean']:
                ds_sMean[dataset] = get_sMean(da_vert)          # dims: (plev, time)


        # ------------------------------------------------------------------------------------------- Plot -------------------------------------------------------------------------------------------------- #
        if switch_test['tMean']:
            ds = ds_tMean
            label = f'{var_name}{region}'
            vmin = None
            vmax = None
            cmap = 'Blues'
            filename = f'a_{var_name}{region}_tMean.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

        if switch_test['zMean']:
            ds = ds_tMean
            label = f'{var_name}{region}'
            vmin = None
            vmax = None
            cmap = 'Blues'
            filename = f'c_{var_name}{region}_zMean.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

        if switch_test['ztMean']:
            if 'plev' in da.dims:
                da = ds_ztMean[dataset]
                fig = plt.figure(figsize=(10, 6))
                da.plot.contourf(x='lat', y='plev', levels=20, cmap='viridis')
                plt.gca().invert_yaxis()  # Invert y-axis to have pressure levels in descending order from top to bottom
                plt.title('mse vertical profile')
                plt.xlabel('Latitude')
                plt.ylabel('Pressure Level (hPa)')
                filename = f'{var_name}vertical_profile.png'
                sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

        if switch_test['sMean']:
            ds = ds_tMean
            label = f'{var_name}{region}'
            vmin = None
            vmax = None
            cmap = 'Blues'
            filename = f'b_{var_name}{region}_sMean.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')



