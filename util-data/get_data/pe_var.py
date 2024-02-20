''' 
# --------------------------------
#  Pe - Precipitation efficiency
# --------------------------------
Designing precipitation efficiency metric:
The fraction of column-integrated condensate to surface precipitation
pe = pr / clwvi

pe      - Precipitation efficiency
pr      - Surface precipitation                         (mm/day)    (kg/m^2)
clwvi   - Column-integrated condensate (liquid + ice)               (kg/m^2)
'''



# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_base as vB
import get_data.missing_data as mD




def get_pe(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Precipitation efficiency 
    Removes small clwvi values to avoid unrealistic pe-values'''
    # switch = {'ocean_mask': True}
    pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily').resample(time='1MS').mean(dim='time')    
    # threshold = 0.99
    # value_threshold = pr.quantile(threshold, dim=('lat', 'lon', 'time')) 
    # pr = pr.where(pr < value_threshold, np.nan)     

    clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution, timescale = 'monthly').resample(time='1MS').mean(dim='time')                   
    # threshold = 0.10 #0.40 #0.10 #0.01 #0.10 #0.20 #0.15 #0.05 #0.01  #[1, 0.20]
    threshold = 0.01 #0.40 #01
    value_threshold = clwvi.quantile(threshold, dim=('lat', 'lon', 'time')) 
    clwvi = clwvi.where(clwvi > value_threshold, np.nan)     

    da = pr / clwvi
    return da

def calc_freq_occur(y):
    ymin = np.nanmin(y)
    ymax = np.nanmax(y)
    edges = np.linspace(ymin, ymax, 101)
    y_bins = []
    for i in range(len(edges)-1):
        if i == len(edges)-2:
            y_value = (xr.where((y>=edges[i]) & (y<=edges[i+1]), 1, 0).sum() / len(y))*100
        else:
            y_value = (xr.where((y>=edges[i]) & (y<edges[i+1]), 1, 0).sum() / len(y))*100
        y_bins.append(y_value)
    bins_middle = edges[0:-1] + (edges[1] - edges[0])/2
    return bins_middle, y_bins



if __name__ == '__main__':
    print('pe testing starting')
    import xarray as xr

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import metric_data as mDd

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import line_plot as lP
    import map_plot as mP
    import scatter_label as sL
    import scatter_time as sT

    sys.path.insert(0, f'{os.getcwd()}/util-calc')
    import ls_state.means_calc as mC


    switch = {'ocean_mask': True}

    switch_test = {                 # can only do one at a time currently
        'delete_previous_plots':        True,
        'check NaN':                    False,
        'plot_scene':                   False,
        'freq_occur':                   False,
        'percentiles':                  False,
        'mean_plot':                    False,
        'sMean':                        True
        }


    
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
    ds_x = xr.Dataset()
    ds = xr.Dataset()
    ymin, ymax = [], []
    for dataset, experiment in mD.run_dataset(var = 'pe', datasets = cD.datasets, experiments = cD.experiments):
# ------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = 'daily').resample(time='1MS').mean(dim='time')    
        clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution = cD.resolutions[0], timescale = 'monthly').resample(time='1MS').mean(dim='time')     
        pe = get_pe(switch = switch, var_name = '', dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'monthly')  
        rome = mDd.load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = experiment).resample(time='1MS').mean(dim='time')     
        # print(pr)
        # print(clwvi)
        # print(pe)
        # print(rome)
        # exit()


# -------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
        if switch_test['check NaN']:
            da = clwvi
            print(f'Number of NaN: {np.sum(np.isnan(da)).compute().data}')
            print(f'min value: {np.min(da).compute().data}')
            print(f'max value: {np.max(da).compute().data}')
            # KACE has a bunch of NaN (10 000 - 13 500), the rest have no NaN
            # IITM-ESM has clwvi in grams instead of Kg (adjusted in cmip_data.py)

        if switch_test['plot_scene']:
            # da = clwvi.isel(time = 0)
            # ds[dataset] = da
            # filename = f'{dataset}_clwvi.png'
            # fig, ax = mP.plot_dsScenes(ds, label = 'units []', title = '', vmin = None, vmax = None, cmap = 'Blues', variable_list = [dataset])
            # mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

            # da = pr.isel(time = 0)
            # ds[dataset] = da
            # filename = f'{dataset}_pr.png'
            # fig, ax = mP.plot_dsScenes(ds, label = 'units []', title = '', vmin = None, vmax = None, cmap = 'Blues', variable_list = [dataset])
            # mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            # high values of precipitation and integraded condensate is not always colocated (could do pe in convective regions)

            da = pe.mean(dim = 'time') #isel(time = 0)
            ds[dataset] = da
            # vmin = 0
            # vmax = 250
            # filename = f'{dataset}_clwvi.png'
            # fig, ax = mP.plot_dsScenes(ds, label = 'units []', title = '', vmin = vmin, vmax = vmax, cmap = 'Blues', variable_list = [dataset])
            # mP.show_plot(fig, show_type = 'save_cwd', filename = filename)


        if switch_test['freq_occur']:
            # da = clwvi
            _, y_bins = calc_freq_occur(da.values.flatten())
            bins = np.arange(1,101)
            ds[dataset] = xr.DataArray(y_bins)
            ymin.append(np.min(y_bins))
            ymax.append(np.max(y_bins))

        if switch_test['percentiles']:
            # da = clwvi
            da = pe
            x = np.arange(1, 101, 1)
            y = da.quantile(x/100, dim=('time', 'lat', 'lon')).compute()
            ds[dataset] = y
            ymin.append(np.min(y))
            ymax.append(np.max(y))

        if switch_test['sMean']:
            da = rome
            current_length = da.sizes['time']
            da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds_x[dataset] = da

            da = pe
            da_sMean = mC.get_sMean(da)
            current_length = da_sMean.sizes['time']
            da_sMean = xr.DataArray(da_sMean.data, dims=['time'], coords={'time': np.arange(0, current_length)})
            ds[dataset] = da_sMean


# ------------------------------------------------------------------------------------ plot datasets together --------------------------------------------------------------------------------------------------- #
    # print(ds_x)
    # print(ds)
    # exit()

    if switch_test['plot_scene']:
        vmin = 0
        vmax = 250
        filename = f'{dataset}_clwvi.png'
        fig, ax = mP.plot_dsScenes(ds, label = 'units []', title = '', vmin = vmin, vmax = vmax, cmap = 'Blues', variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['freq_occur']:
        ymin = np.min(ymin)
        ymax = np.max(ymax)
        label_x = 'fraction of total range [%]'
        label_y = 'Freq. occur [%]'
        filename = 'value_distribution_fractional_values.png'
        colors = lP.generate_distinct_colors(len(ds.data_vars))
        fig, axes = lP.plot_dsLine(ds, x = bins, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x, label_y = label_y, colors = colors, 
                    ymin = ymin, ymax = ymax,
                    fig_given = False, one_ax = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['percentiles']:
        ymin = np.min(ymin)
        ymax = np.max(ymax)
        # ds = ds.sel(quantile=slice(0, 0.99))
        # x = np.arange(1, 100, 1)
        # ymin = None 
        # ymax = None 
        label_x = ''
        label_y = ''
        filename = 'value_distribution_percentiles.png'
        colors = lP.generate_distinct_colors(len(ds.data_vars))
        fig, axes = lP.plot_dsLine(ds, x = x, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x, label_y = label_y, colors = colors, 
                    ymin = ymin, ymax = ymax,
                    fig_given = False, one_ax = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
        # IITM-ESM has a strange distribution
        # for variable in list(ds.data_vars.keys()):
        #     print(ds[variable].sel(quantile = 0.05).data)


    if switch_test['sMean']:
        plot = False # monthly correlation
        if plot:
            filename = 'basic.png'
            fig_title = 'cmip6 precipitation efficiency with organization'
            label_x = r'rome [km${^2}$]'
            label_y = r'pe [day${^-1}$]'
            label_cmap = 'months [nb]'
            vmin = 0 
            vmax = 10
            fig, axes = sT.plot_dsScatter(ds_x = ds_x, ds_y = ds, variable_list = list(ds_x.data_vars.keys()), 
                                        fig_title = fig_title, label_x = label_x, label_y = label_y, label_cmap = label_cmap,
                                        colors_scatter = ['k']*len(ds_x.data_vars), colors_slope = ['k']*len(ds_x.data_vars), cmap = 'Blues',
                                        xmin = None, xmax = None, ymin = None, ymax = None, vmin = vmin, vmax = vmax,
                                        density_map = True, models_highlight = ['a', 'b'],
                                        fig_given = False, ax = '', individual_cmap = False)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        plot = True # time-mean correlation
        if plot:
            filename = 'scatter_label.png'
            fig_title = 'cmip6 precipitation efficiency with organization'
            label_x = r'rome [km${^2}$]'
            label_y = r'pe [day${^-1}$]'
            fig, ax = sL.plot_dsScatter(ds_x = ds_x.mean(dim = 'time'), ds_y = ds.mean(dim = 'time'), variable_list = list(ds_x.data_vars.keys()), fig_title = 'test', x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

















