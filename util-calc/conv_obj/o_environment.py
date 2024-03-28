'''
# -----------
#   in_obj
# -----------
Calculates property in convective objects
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np


# ----------------------------------------------------------------------------------- imported scripts ----------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_calc as vC
import var_calc.conv_obj_var as cO

sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc      as mean_calc



# ---------------------------
#   Mask by object location
# ---------------------------
# -------------------------------------------------------------------------------------- mask (daily)----------------------------------------------------------------------------------------------- #
def get_in_conv_daily(da, conv_obj):
    ''' Convective objects are defined on daily timescales, so here we mask the variable with the object location'''
    da_conv = da.where(conv_obj > 0, np.nan)
    return da_conv


# -------------------------------------------------------------------------------------- mask (monthly) ----------------------------------------------------------------------------------------------- #
def get_conv_freq(dataset, experiment, switch = {}, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    ''' Number of times a gridbox is convective per month '''
    conv_obj, region = vC.get_variable({'conv_obj':True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
    if cD.timescales[0] == 'daily':
        conv_freq = xr.where(conv_obj>0, 1, np.nan)
    elif cD.timescales[0] == 'monthly':
        conv = xr.where(conv_obj>0, 1, np.nan)
        conv_freq = conv.resample(time='1MS').sum(dim='time')    
    return conv_freq

def get_in_conv_monthly(da, conv_obj):
    ''' To mask variables on monthly timescales:
    Takes the gridboxes convered by convection for a month, and weight the gridbox contribution to the mean by the frequency of occurence of convection in the gridbox'''
    conv = xr.where(conv_obj > 0, 1, np.nan)
    conv = conv.resample(time='1MS').mean(dim='time')    
    da = da.resample(time='1MS').mean(dim='time')    
    da_convFreq_weighted = da * conv
    return da_convFreq_weighted


# -------------------------------------------------------------------------------------- summarize ----------------------------------------------------------------------------------------------- #
def get_in_conv(da, dataset, experiment, switch = {}, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    ''' Masks variable with the location of objects, and calculates the spatial mean'''
    conv_obj, region = vC.get_variable({'conv_obj':True}, switch, dataset, experiment, resolution, timescale = 'daily', from_folder = True, re_process = False)
    da_conv = get_in_conv_daily(da, conv_obj)    if timescale == 'daily'     else None
    da_conv = get_in_conv_monthly(da, conv_obj)  if timescale == 'monthly'   else da
    return da_conv

def get_in_conv_sMean(da, dataset, experiment, switch = {}, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    ''' Masks variable with the location of objects, and calculates the spatial mean'''
    da_conv = get_in_conv(da, dataset, experiment, switch, resolution, timescale)
    da_conv_sMean = mean_calc.get_sMean(da_conv)
    return da_conv_sMean



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import dimensions_data  as dD
    import missing_data     as mD
    import metric_data as mDd

    sys.path.insert(0, f'{os.getcwd()}/util-calc')
    import ls_state.means_calc      as mean_calc

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot        as mP
    import get_plot.line_plot       as lP
    import get_plot.scatter_time    as sT
    import get_plot.scatter_label   as sL

    import xarray as xr

    switch = {
        'test_sample': True, 'ocean_mask': True
        }

    switch_var = {
        'pr': False,
        'pe': True
        }
    
    switch_test ={
        'delete_previous_plots':                    True,   
        'plot_mask':                                False,  # 1
        'plot_variable':                            False,  # 2
        'plot_variable_masked':                     False,  # 3
        'sMean':                                    False,  # 4
        'scatter_w_org':                            False,  # 5
        'scatter_w_org_tMean_historical':           False,  # 6
        'scatter_w_org_tMean_change_with_warming':  True,  # 7
        }

    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None

    experiment = cD.experiments[0]
    var_name = next((key for key, value in switch_var.items() if value), None)
    print(f'variable: {var_name}')
    
    ds_mask = xr.Dataset()
    ds_variable = xr.Dataset()
    ds_variable_masked = xr.Dataset()
    ds_variable_masked_sMean = xr.Dataset()
    ds_tas, ds_tas_warm = xr.Dataset(), xr.Dataset()
    ds_x_sMean, ds_x_sMean_warm = xr.Dataset(), xr.Dataset()
    ds_y_sMean, ds_y_sMean_warm = xr.Dataset(), xr.Dataset()

    for dataset in mD.run_dataset_only(var = 'pe', datasets = cD.datasets):
    # ------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        print(f'dataset: {dataset}')
        da, region = vC.get_variable(switch_var, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
        da_conv = get_in_conv(da, dataset, experiment, switch)
        # print(da)
        # print(da_conv)

    # ------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
        if switch_test['plot_mask']:
            conv_freq = get_conv_freq(dataset, experiment, switch = switch, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            ds_mask[dataset] = conv_freq.isel(time=0)

        if switch_test['plot_variable']:
            ds_variable[dataset] = da.isel(time = 0)

        if switch_test['plot_variable_masked']:
            da_conv = get_in_conv(da, dataset, experiment, switch, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            ds_variable_masked[dataset] = da_conv.isel(time=0)

        if switch_test['sMean']:
            da_conv_sMean = get_in_conv_sMean(da, dataset, experiment, switch = switch, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            current_length = da_conv_sMean.sizes['time']
            da_conv_sMean = xr.DataArray(da_conv_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_variable_masked_sMean[dataset] = da_conv_sMean

        if switch_test['scatter_w_org_tMean_historical']:
            # historical calc
            da = rome = mDd.load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = experiment, timescale = 'daily').resample(time='1MS').mean(dim='time')    
            current_length = da.sizes['time']
            da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_x_sMean[dataset] = da
            da, _ = vC.get_variable({var_name: True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0]) 
            da_sMean = get_in_conv_sMean(da, dataset, experiment, switch = switch, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            current_length = da_sMean.sizes['time']
            da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_y_sMean[dataset] = da_sMean

            ds_tas[dataset] = mDd.load_metric(metric_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = experiment, timescale = 'monthly').resample(time='1MS').mean(dim='time').mean(dim = 'time')

        if switch_test['scatter_w_org_tMean_change_with_warming']:
            # historical calc
            da = rome = mDd.load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = experiment, timescale = 'daily').resample(time='1MS').mean(dim='time')    
            current_length = da.sizes['time']
            da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_x_sMean[dataset] = da
            da, _ = vC.get_variable({var_name: True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0]) 
            da_sMean = get_in_conv_sMean(da, dataset, experiment, switch = switch, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            current_length = da_sMean.sizes['time']
            da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_y_sMean[dataset] = da_sMean

            ds_tas[dataset] = mDd.load_metric(metric_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = experiment, timescale = 'monthly').resample(time='1MS').mean(dim='time').mean(dim = 'time')

            # warm calc
            rome_warm = mDd.load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = 'ssp585', timescale = 'daily').resample(time='1MS').mean(dim='time')     
            da = rome_warm
            current_length = da.sizes['time']
            da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_x_sMean_warm[dataset] = da

            da, _ = vC.get_variable({var_name: True}, switch, dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = cD.timescales[0])
            da_sMean = get_in_conv_sMean(da, dataset, experiment= 'ssp585', switch = switch, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            current_length = da_sMean.sizes['time']
            da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_y_sMean_warm[dataset] = da_sMean

            tas_warm = mDd.load_metric(metric_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = 'ssp585', timescale = 'monthly').resample(time='1MS').mean(dim='time').mean(dim = 'time')   

            ds_tas_warm[dataset] = tas_warm


    # ----------------------------------------------------------------------------------------- plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_mask']:
        label = '[Nb]'
        vmin = None #0
        vmax = None #250
        cmap = 'Blues'
        title = f'object_frequency_{dataset}'
        filename = f'{title}.png'
        fig, ax = mP.plot_dsScenes(ds_mask, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds_mask.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_variable']:
        if switch_var['pe']:
            label = 'kg'
            vmin = 0
            vmax = 250
            cmap = 'Blues'
        if switch_var['pr']:
            label = '[mm/day]'
            vmin = None
            vmax = None
            cmap = 'Blues'
        filename = f'{var_name}{region}_{dataset}.png'
        fig, ax = mP.plot_dsScenes(ds_variable, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds_variable.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_variable_masked']:
        if switch_var['pe']:
            label = 'kg'
            vmin = 0
            vmax = 250
            cmap = 'Blues'
        if switch_var['pr']:
            label = '[mm/day]'
            vmin = None
            vmax = None
            cmap = 'Blues'
        filename = f'{var_name}{region}_masked.png'
        fig, ax = mP.plot_dsScenes(ds_variable_masked, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds_variable_masked.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['sMean']:
        label_x = f'time [{cD.timescales[0]}]'
        if switch_var['pe']:
            label_y = 'kg'
            ymin = 0
            ymax = 250
            cmap = 'Blues'
        if switch_var['pr']:
            label_y = '[mm/day]'
            vmin = None
            vmax = None
            cmap = 'Blues'
        filename = f'{var_name}{region}_masked_sMean.png'
        fig, axes = lP.plot_dsLine(ds_variable_masked_sMean, x = None, variable_list = list(ds_variable_masked_sMean.data_vars.keys()), title = '', label_x = '', label_y = '', colors = ['k']*len(ds_variable_masked_sMean.data_vars), 
                xmin = None, xmax = None, ymin = ymin, ymax = ymax,
                fig_given = False, one_ax = True, fig = '', axes = '',
                distinct_colors = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['scatter_w_org']:
        filename = f'scatter_time_{var_name}{region}_in_obj.png'
        label_cmap = 'months [nb]'
        vmin = 0 
        vmax = 10
        label_x_sMean = r'rome [km${^2}$]' 
        label_y_sMean = r'pe [day${^-1}$]'      if var_name == 'pe'      else None
        fig, axes = sT.plot_dsScatter(ds_x = ds_x_sMean, ds_y = ds_y_sMean, variable_list = list(ds_x_sMean.data_vars.keys()), 
                                    fig_title = filename, label_x = label_x_sMean, label_y = label_y_sMean, label_cmap = label_cmap,
                                    colors_scatter = ['k']*len(ds_x_sMean.data_vars), colors_slope = ['k']*len(ds_x_sMean.data_vars), cmap = 'Blues',
                                    xmin = None, xmax = None, ymin = None, ymax = None, vmin = vmin, vmax = vmax,
                                    density_map = True, models_highlight = ['a', 'b'],
                                    fig_given = False, ax = '', individual_cmap = False)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['scatter_w_org_tMean_historical']:
        label_x = r'rome [km$^{2}$]'
        label_y = r'pe [day$^{-1}$]'
        filename = f'{var_name}{region}_rome_historical.png'
        x = ds_x_sMean.mean(dim = 'time')
        y = ds_y_sMean.mean(dim = 'time')
        fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = [''], color_highlight = 'b', 
                    add_correlation = True, put_point = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['scatter_w_org_tMean_change_with_warming']:
        label_x = r'rome [km$^{2}$K$^{-1}$]'
        label_y = r'pe [day$^{-1}$K$^{-1}$]'
        filename = f'{var_name}{region}_rome_change_with_warming.png'
        x = (ds_x_sMean_warm.mean(dim = 'time') - ds_x_sMean.mean(dim = 'time')) / (ds_tas_warm - ds_tas)
        y = (ds_y_sMean_warm.mean(dim = 'time') - ds_y_sMean.mean(dim = 'time')) / (ds_tas_warm - ds_tas)
        fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                    xmin = None, xmax = None, ymin = None, ymax = None,
                    fig_given = False, fig = '', ax = '', 
                    color = 'k', models_highlight = [''], color_highlight = 'b', 
                    add_correlation = True, put_point = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)








        


