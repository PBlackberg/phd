''' 
# --------------------------------
#  Pe - Precipitation efficiency
# --------------------------------
Calculates precipitation efficiency (pe)
Function:
    pe = get_pe(switch, var_name, dataset, experiment, resolution, timescale)

Input:
    pr      - surface precipitation                         dim: (time, lat, lon)   get from: util-data/variable_base.py
    clwvi   - Column-integrated condensate (liquid + ice)   dim: (time, lat, lon)   get from: util-data/variable_base.py

Output:
    pe:     - list                                          dim: (time)
'''



# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np


# ------------------------------------------------------------------------------------- imported scripts ---------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_base as vB

sys.path.insert(0, f'{os.getcwd()}/util-calc')
import statistics_calc.means_calc as mC



# ------------------------
#   Calculate variable
# ------------------------
def get_pe(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Precipitation efficiency (Fraction of column-integrated condensate to surface precipitation)
    pe = pr / clwvi
    Where
        pr      - Surface precipitation                         (mm/day) or equivalently (kg/m^2)
        clwvi   - Column-integrated condensate (liquid + ice)   (kg/m^2) '''
    pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily').resample(time='1MS').mean(dim='time')    
    # threshold = 0.99
    # value_threshold = pr.quantile(threshold, dim=('lat', 'lon', 'time')) 
    # pr = pr.where(pr < value_threshold, np.nan)     
    clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution, timescale = 'monthly').resample(time='1MS').mean(dim='time')                   
    threshold = 0.01 #0.10 #0.20
    value_threshold = clwvi.quantile(threshold, dim=('lat', 'lon', 'time')) 
    clwvi = clwvi.where(clwvi > value_threshold, np.nan)     
    da = pr / clwvi
    return da

def get_pe_mean_first(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Precipitation efficiency 
    Averages clwvi and pr first before calculating the quotient '''
    pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily').resample(time='1MS').mean(dim='time')    
    pr_sMean = mC.get_sMean(pr)
    # threshold = 0.99
    # value_threshold = pr.quantile(threshold, dim=('lat', 'lon', 'time')) 
    # pr = pr.where(pr < value_threshold, np.nan)     
    clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution, timescale = 'monthly').resample(time='1MS').mean(dim='time')                   
    threshold = 0.01 #0.10 #0.20
    value_threshold = clwvi.quantile(threshold, dim=('lat', 'lon', 'time')) 
    clwvi = clwvi.where(clwvi > value_threshold, np.nan)     
    clwvi_sMean = mC.get_sMean(clwvi)
    da = pr_sMean / clwvi_sMean
    return da

def get_pe_2(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Precipitation efficiency 
    Removes small clwvi values to avoid unrealistic pe-values'''
    pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily').resample(time='1MS').mean(dim='time')    
    pr = pr.where(pr > 3, np.nan)     
    clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution, timescale = 'monthly').resample(time='1MS').mean(dim='time')                   
    threshold = 0.01 #0.10 #0.20
    value_threshold = clwvi.quantile(threshold, dim=('lat', 'lon', 'time')) 
    clwvi = clwvi.where(clwvi > value_threshold, np.nan)     
    da = pr / clwvi
    return da


# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('pe test starting')
    import xarray as xr

    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data as mD
    import metric_data as mDd

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.line_plot        as lP
    import get_plot.map_plot         as mP
    import get_plot.scatter_label    as sL
    import get_plot.scatter_time     as sT


    switch = {'ocean_mask': False}

    switch_var = {
        'clwvi':    False,
        'pr':       False,
        'pe':       True
        }

    switch_test = {
        'delete_previous_plots':        True,
        'check NaN':                    False,
        'plot_snapshot':                True,
        'plot_tMean':                   True,
        'percentiles':                  True,
        'sMean':                        True,
        'mean':                         True,
        'mean_meanFirst':               False    # taking the spatial mean before quotient calc
        }

    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None

    experiment = cD.experiments[0]
    
    for var_name in [k for k, v in switch_var.items() if v]:
        print(f'for: {var_name}')
        ds_snapshot, ds_tMean, ds_tMean_warm = xr.Dataset(), xr.Dataset(), xr.Dataset()
        ds_x_percentiles, ds_percentile_values = xr.Dataset(), xr.Dataset()
        ds_x_sMean, ds_y_sMean = xr.Dataset(), xr.Dataset()
        ds_x_sMean_warm, ds_y_sMean_warm = xr.Dataset(), xr.Dataset()
        ds_tas, ds_tas_warm = xr.Dataset(), xr.Dataset()
        ds_x_sMean_meanFirst, ds_x_sMean_meanFirst_warm = xr.Dataset(), xr.Dataset()
        ds_y_sMean_meanFirst, ds_y_sMean_meanFirst_warm = xr.Dataset(), xr.Dataset()

        for dataset in mD.run_dataset_only(var = 'pe', datasets = cD.datasets):
            # ------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
            pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = 'daily').resample(time='1MS').mean(dim='time')    
            clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')     
            rome = mDd.load_metric(met_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = experiment, timescale = 'daily').resample(time='1MS').mean(dim='time')     
            tas = mDd.load_metric(met_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = experiment, timescale = 'monthly').resample(time='1MS').mean(dim='time').mean(dim = 'time')     
            pe = get_pe(switch = switch, var_name = 'pe', dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')
            pe_mean_first = get_pe_mean_first(switch = switch, var_name = 'pe', dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')
            # print(pr)
            # print(clwvi)
            # print(pe)
            # print(rome)
            # print(tas)
            # exit()


            # -------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
            if switch_test['check NaN']:
                da = clwvi  if var_name == 'clwvi'  else None
                da = pr     if var_name == 'pr'     else da
                da = pe     if var_name == 'pe'     else da

            if switch_test['plot_snapshot']:
                ds_snapshot[dataset] = clwvi.isel(time = 0) if var_name == 'clwvi'  else None
                ds_snapshot[dataset] = pr.isel(time = 0)    if var_name == 'pr'     else ds_snapshot[dataset]              
                ds_snapshot[dataset] = pe.isel(time = 0)    if var_name == 'pe'     else ds_snapshot[dataset] 

            if switch_test['plot_tMean']:
                    ds_tMean[dataset] = clwvi.mean(dim = 'time')    if var_name == 'clwvi'  else None
                    ds_tMean[dataset] = pr.mean(dim = 'time')       if var_name == 'pr'     else ds_tMean[dataset] 
                    ds_tMean[dataset] = pe.mean(dim = 'time')       if var_name == 'pe'     else ds_tMean[dataset] 

            if switch_test['percentiles']:
                da = clwvi  if var_name == 'clwvi'  else None
                da = pr     if var_name == 'pr'     else da
                da = pe     if var_name == 'pe'     else da
                percentiles = np.arange(1, 101, 1)
                ds_percentile_values[dataset] = da.quantile(percentiles/100, dim=('time', 'lat', 'lon')).compute()

            if switch_test['sMean'] or switch_test['mean']:
                # historical calc
                da = rome
                current_length = da.sizes['time']
                da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_x_sMean[dataset] = da
                da = clwvi  if var_name == 'clwvi'  else None
                da = pr     if var_name == 'pr'     else da
                da = pe     if var_name == 'pe'     else da
                da_sMean = mC.get_sMean(da)
                current_length = da_sMean.sizes['time']
                da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_y_sMean[dataset] = da_sMean

                ds_tas[dataset] = tas

                # warm calc
                rome_warm = mDd.load_metric(met_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = 'ssp585', timescale = 'daily').resample(time='1MS').mean(dim='time')     
                da = rome_warm
                current_length = da.sizes['time']
                da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_x_sMean_warm[dataset] = da

                da = vB.load_variable({'clwvi': True}, switch, dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')             if var_name == 'clwvi'  else None
                da = vB.load_variable({'pr': True}, switch, dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'daily').resample(time='1MS').mean(dim='time')                  if var_name == 'pr'     else da
                da = get_pe(switch = switch, var_name = 'pe', dataset = dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')    if var_name == 'pe'     else da
                da_sMean = mC.get_sMean(da)
                current_length = da_sMean.sizes['time']
                da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_y_sMean_warm[dataset] = da_sMean

                tas_warm = mDd.load_metric(met_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = 'ssp585', timescale = 'monthly').resample(time='1MS').mean(dim='time').mean(dim = 'time')   

                ds_tas_warm[dataset] = tas_warm

            if switch_test['mean_meanFirst'] and var_name == 'pe':
                # historical calc
                da = rome
                current_length = da.sizes['time']
                da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_x_sMean_meanFirst[dataset] = da

                da = clwvi  if var_name == 'clwvi'  else None
                da = pr     if var_name == 'pr'     else da
                da_sMean = pe_mean_first      if var_name == 'pe'     else da
                current_length = da_sMean.sizes['time']
                da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_y_sMean_meanFirst[dataset] = da_sMean
                ds_tas[dataset] = tas

                # warm calc
                rome_warm = mDd.load_metric(met_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = 'ssp585', timescale = 'daily').resample(time='1MS').mean(dim='time')     
                da = rome_warm
                current_length = da.sizes['time']
                da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_x_sMean_meanFirst_warm[dataset] = da

                da_sMean = vB.load_variable({'clwvi': True}, switch, dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')             if var_name == 'clwvi'  else None
                da_sMean = vB.load_variable({'pr': True}, switch, dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'daily').resample(time='1MS').mean(dim='time')                  if var_name == 'pr'     else da
                da_sMean = get_pe_mean_first(switch = switch, var_name = 'pe', dataset = dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')    if var_name == 'pe'     else da
                current_length = da_sMean.sizes['time']
                da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
                ds_y_sMean_meanFirst_warm[dataset] = da_sMean
                tas_warm = mDd.load_metric(met_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = 'ssp585', timescale = 'monthly').resample(time='1MS').mean(dim='time').mean(dim = 'time')   
                ds_tas_warm[dataset] = tas_warm


        # ------------------------------------------------------------------------------------ plot datasets together --------------------------------------------------------------------------------------------------- #
        if switch_test['check NaN']:
            print(f'Number of NaN: {np.sum(np.isnan(da)).compute().data}')
            print(f'min value: {np.min(da).compute().data}')
            print(f'max value: {np.max(da).compute().data}')
            # KACE has a bunch of NaN (10 000 - 13 500), the rest have no NaN   (clwvi) (505 with no ocean)
            # IITM-ESM has clwvi in grams instead of Kg                         (adjusted in cmip_data.py)

        if switch_test['plot_snapshot']:
            if var_name == 'clwvi':
                vmin_snapshot = 0
                vmax_snapshot = 0.3
                label_snapshot = '[kg]'
                title_snapshot = 'snapshot_clwvi'
            if var_name == 'pr':
                vmin_snapshot = 0
                vmax_snapshot = 10
                label_snapshot = '[mm/day]'
                title_snapshot = 'snapshot_pr'
            if var_name == 'pe':
                vmin_snapshot = 0
                vmax_snapshot = 300
                label_snapshot = '[/day]'
                title_snapshot = 'snapshot_pe'
            filename = f'cmip6_{title_snapshot}.png'
            filename = f'cmip6_ocean_{title_snapshot}.png' if switch['ocean_mask'] else filename
            fig, ax = mP.plot_dsScenes(ds_snapshot, label = label_snapshot, title = filename, vmin = vmin_snapshot, vmax = vmax_snapshot, cmap = 'Blues', variable_list = list(ds_snapshot.data_vars.keys()))
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            # high values of precipitation and integrated condensate is not always colocated (could do pe in convective regions)

        if switch_test['plot_tMean']:
            if var_name == 'clwvi':
                vmin_tMean = 0
                vmax_tMean = 0.3
                label_tMean = '[kg]'
                title_tMean = 'tMean_clwvi'
            if var_name == 'pr':
                vmin_tMean = 0
                vmax_tMean = 10
                label_tMean = '[mm/day]'
                title_tMean = 'tMean_pr'
            if var_name == 'pe':
                vmin_tMean = 0
                vmax_tMean = 300
                label_tMean = '[/day]'
                title_tMean = 'tMean_pe'
            filename = f'cmip6_{title_tMean}.png'
            filename = f'cmip6_ocean_{title_tMean}.png' if switch['ocean_mask'] else filename
            fig, ax = mP.plot_dsScenes(ds_tMean, label = label_tMean, title = filename, vmin = vmin_tMean, vmax = vmax_tMean, cmap = 'Blues', variable_list = list(ds_tMean.data_vars.keys()))
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['percentiles']:
            # print(ds_percentile_values)
            # exit()
            percentile_range = [0, 1]
            ds = ds_percentile_values.sel(quantile=slice(percentile_range[0], percentile_range[1]))
            ymin_prc_list, ymax_prc_list = [], []
            for variable in ds.data_vars:
                ymin_prc_list.append(ds[variable].min())
                ymax_prc_list.append(ds[variable].max())
            ymin = min(ymin_prc_list)
            ymax = max(ymax_prc_list)
            x = ds['quantile'] * 100
            title_percentile = f'{var_name}_value_distribution_percentiles'
            label_y_percentile = 'percentile value'
            label_x_percentile = 'percentile'
            filename = f'cmip6_{title_percentile}.png'
            filename = f'cmip6_ocean_{title_percentile}.png' if switch['ocean_mask'] else filename
            colors = lP.generate_distinct_colors(len(ds.data_vars))
            fig, axes = lP.plot_dsLine(ds, x = x, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x_percentile, label_y = label_y_percentile, colors = colors, 
                        ymin = ymin, ymax = ymax,
                        fig_given = False, one_ax = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            # IITM-ESM has a strange distribution
            # for variable in list(ds.data_vars.keys()):
            # print(ds[variable].sel(quantile = 0.05).data)

        if switch_test['sMean']:
            filename = f'cmip6_{var_name}_inter-month.png'
            filename = f'cmip6_ocean_{var_name}_inter-month.png' if switch['ocean_mask'] else filename
            label_cmap = 'months [nb]'
            vmin = 0 
            vmax = 10
            label_x_sMean = r'rome [km${^2}$]' 
            label_y_sMean = r'clwvi [kg$]'          if var_name == 'clwvi'   else None
            label_y_sMean = r'pr [mm day${^-1}$]'   if var_name == 'pr'      else label_y_sMean
            label_y_sMean = r'pe [day${^-1}$]'      if var_name == 'pe'      else label_y_sMean
            fig, axes = sT.plot_dsScatter(ds_x = ds_x_sMean, ds_y = ds_y_sMean, variable_list = list(ds_x_sMean.data_vars.keys()), 
                                        fig_title = filename, label_x = label_x_sMean, label_y = label_y_sMean, label_cmap = label_cmap,
                                        colors_scatter = ['k']*len(ds_x_sMean.data_vars), colors_slope = ['k']*len(ds_x_sMean.data_vars), cmap = 'Blues',
                                        xmin = None, xmax = None, ymin = None, ymax = None, vmin = vmin, vmax = vmax,
                                        density_map = True, models_highlight = ['a', 'b'],
                                        fig_given = False, ax = '', individual_cmap = False)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['mean']:
            # historical
            label_x = r'rome [km${^2}$]'
            label_y = r'pe [day${^-1}$]'
            filename = f'{var_name}_rome_historical.png'
            x = ds_x_sMean.mean(dim = 'time')
            y = ds_y_sMean.mean(dim = 'time')
            fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

            # Change with warming
            label_x = r'rome [km${^2}$K${^-1}$]'
            label_y = r'pe [day${^-1}$K${^-1}$]'
            filename = f'{var_name}_rome_change_with_warming.png'
            x = (ds_x_sMean_warm.mean(dim = 'time') - ds_x_sMean.mean(dim = 'time')) / (ds_tas_warm - ds_tas)
            y = (ds_y_sMean_warm.mean(dim = 'time') - ds_y_sMean.mean(dim = 'time')) / (ds_tas_warm - ds_tas)
            fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['mean_meanFirst'] and var_name == 'pe':
            # historical
            label_x = r'rome [km${^2}$]'
            label_y = r'pe [day${^-1}$]'
            filename = f'{var_name}_mean_first_rome_historical.png'
            x = ds_x_sMean.mean(dim = 'time')
            y = ds_y_sMean.mean(dim = 'time')
            fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

            # Change with warming
            label_x = r'rome [km${^2}$K${^-1}$]'
            label_y = r'pe [day${^-1}$K${^-1}$]'
            filename = f'{var_name}_mean_first_rome_change_with_warming.png'
            x = (ds_x_sMean_warm.mean(dim = 'time') - ds_x_sMean.mean(dim = 'time')) / (ds_tas_warm - ds_tas)
            y = (ds_y_sMean_warm.mean(dim = 'time') - ds_y_sMean.mean(dim = 'time')) / (ds_tas_warm - ds_tas)
            fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)



    # ----------------------------
    #  Call from different script
    # ----------------------------
    # to use:
    # import os
    # import sys
    # sys.path.insert(0, f'{os.getcwd()}/util-data')
    # import pe_var as pE
    # pe = pE.get_pe(switch = switch, var_name = 'pe', dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'monthly')  

