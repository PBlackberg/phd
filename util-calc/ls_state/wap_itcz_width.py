'''
# ------------------------
#  ITCZ width calculation
# ------------------------
Based on;
- time-mean vertical pressure velocity at 500 hpa (wap)
- variance in object location (latitude and longitude) (obj)
- Time-mean precipitation distribution (pr)

For latitude variance metric:
If an object close to the equator splits into two, it will be considered to have more objects closer to the equator
(same effect as objects moving closer to the equator)
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np



# ---------------
#  General calc
# ---------------
def calc_variance(aList, reference_point):
    squared_difference = (aList - reference_point)**2
    variance = squared_difference.mean()
    return variance

def calc_mae(aList, reference_point):
    ''' Mean Absolute Error'''
    abs_difference = np.abs(aList - reference_point)
    mae = (abs_difference).mean()
    return mae

def calc_abs_diff(aList, reference_point):
    ''' Absolute Error'''
    abs_difference = np.abs(aList - reference_point)
    return abs_difference

def get_percentile_values(aList, dimensions, percentiles = np.arange(1, 101, 1)):
    return aList.quantile(percentiles/100, dim=dimensions).compute()

def calc_freq_occur(aList, ymin = 0, ymax = 30, nb_bins = 7):
    ''' In units of % '''
    edges = np.linspace(ymin, ymax, nb_bins)
    freq_occur = []
    for i in range(len(edges)-1):
        if i == len(edges)-1:
            freq_occur_bin = (xr.where((aList>=edges[i]) & (aList<=edges[i+1]), 1, 0).sum() / len(aList)).data # include points right at the end of the distribution
        else:
            freq_occur_bin = (xr.where((aList>=edges[i]) & (aList<edges[i+1]), 1, 0).sum() / len(aList)).data
        freq_occur.append(freq_occur_bin)
    bins_middle = edges[0:-1] + (edges[1] - edges[0])/2
    return xr.DataArray(np.array(freq_occur)*100), bins_middle 


# -------------
#  itcz width
# -------------
# -------------------------------------------------------------------------- Based on vertical pressure velocity (wap) ----------------------------------------------------------------------------------------------------- #
def itcz_width_sMean(da):
    da = da.mean(dim = 'lon')
    itcz = xr.where(da < 0, 1, np.nan).compute()                # ascent
    itcz_lats = itcz * da['lat']                                # lats in ascent
    return itcz_lats.max(dim='lat') - itcz_lats.min(dim='lat')  # time series

def itcz_width(da):
    alist = da.mean(dim = ['time', 'lon']).compute()
    itcz_lats = alist.where(alist < 0, drop = True)['lat']
    return itcz_lats.max() - itcz_lats.min()                    # one value

def get_fraction_descent(da, dims):
    da = da.mean(dim = 'time')
    da = ((xr.where(da > 0, 1, 0) * dims.aream).sum() / dims.aream.sum())*100   # fraction of descent (one value)
    return da                              



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('itcz width calc test starting')
    import os
    import sys
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import variable_calc            as vC
    import var_calc.conv_obj_var    as cO
    import missing_data             as mD
    import dimensions_data          as dD
    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.line_plot       as lP
    import get_plot.scatter_time    as sT
    import get_plot.map_plot        as mP
    import get_plot.scatter_label    as sL


    switch_wap = {
        'test_sample':  True, 
        '700hpa':       False,  '500hpa': True,             # for wap
        }

    switch_obj = {
        'test_sample':  True, 
        'fixed_area':   False,
        }

    switch_test = {
        'delete_previous_plots':        True,

        'scene':                        False,
        'itcz_width_sMean':             False,
        'percentiles':                  False,
        'freq_occur':                   False,
        'scatter_org_sMean':            False,

        'tMean':                        False,
        'itcz_width_tMean':             True,
        'scatter_org_tMean':            False,
        'plot_change':                  False,
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None

    experiment = cD.experiments[0]
    
    ds_snapshot = xr.Dataset()
    ds_itcz_sMean = xr.Dataset()
    ds_tMean, ds_tMean_change = xr.Dataset(), xr.Dataset()
    ds_itcz_tMean, ds_itcz_tMean_change = xr.Dataset(), xr.Dataset()
    ds_rome, ds_rome_change = xr.Dataset(), xr.Dataset()
    ds_tas_change = xr.Dataset()
    for dataset in mD.run_dataset_only(var = 'pe', datasets = cD.datasets):
        print(f'dataset: {dataset}')
        # ----------------------------------------------------------------------------------- Get data -------------------------------------------------------------------------------------------------- #
        wap, region = vC.get_variable(switch_var = {'wap': True}, switch = switch_wap, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = False, re_process = False)
        wap_warm, region = vC.get_variable(switch_var = {'wap': True}, switch = switch_wap, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = False, re_process = False)

        # print(f'wap: \n {wap}')
        # exit()


        # ----------------------------------------------------------------------------------- Calculate -------------------------------------------------------------------------------------------------- #
        if switch_test['scene']:
            ds_snapshot[dataset] = wap.isel(time = 0) 

        if switch_test['itcz_width_sMean']:
            ds_itcz_sMean[dataset] = itcz_width_sMean(wap)

        if switch_test['tMean']:
            ds_tMean[dataset] = wap.mean(dim = 'time') 

        if switch_test['itcz_width_tMean']:
            ds_itcz_tMean[dataset] = itcz_width(wap)


        # ------------------------------------------------------------------------------------------ Plot -------------------------------------------------------------------------------------------------- #
        if switch_test['scene']:
            ds = ds_snapshot
            label = 'wap [hpa/day]'
            vmin = None
            vmax = None
            cmap = 'Blues'
            filename = f'wap.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
        
        if switch_test['itcz_width_sMean']:
            # print(ds_itcz_sMean)
            if switch_test['percentiles']:
                ds = xr.Dataset()
                for dataset in ds_itcz_sMean.data_vars:
                    ds[dataset] = get_percentile_values(np.abs(ds_itcz_sMean[dataset]), dimensions = 'obj', percentiles = np.arange(1, 101, 1))
                x = ds['quantile'] * 100
                label_y_percentile = 'percentile value'
                label_x_percentile = 'percentile'
                filename = f'wap_value_distribution_percentiles.png'
                colors = lP.generate_distinct_colors(len(ds.data_vars))
                ymin_prc_list, ymax_prc_list = [], []
                for variable in ds.data_vars:
                    ymin_prc_list.append(ds[variable].min())
                    ymax_prc_list.append(ds[variable].max())
                ymin = min(ymin_prc_list)
                ymax = max(ymax_prc_list)
                fig, axes = lP.plot_dsLine(ds, x = x, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x_percentile, label_y = label_y_percentile, colors = colors, 
                            ymin = ymin, ymax = ymax,
                            fig_given = False, one_ax = True)
                mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            
            if switch_test['freq_occur']:
                ds = xr.Dataset()
                for dataset in ds_itcz_sMean.data_vars:
                    ds[dataset], bins_middle = calc_freq_occur(np.abs(ds_itcz_sMean[dataset]), ymin = 0, ymax = 30, nb_bins = 7)
                label_x = 'latitude [degree]'
                label_y = 'Freq. occur [%]'
                title = 'pe'
                filename = f'wap_freq_occur.png'
                ymin = None
                ymax = None
                colors = lP.generate_distinct_colors(len(ds.data_vars))
                fig, axes = lP.plot_dsLine(ds, x = bins_middle, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x, label_y = label_y, colors = colors, 
                            ymin = ymin, ymax = ymax,
                            fig_given = False, one_ax = True)
                mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['tMean']:
            ds = ds_tMean
            label = 'wap [hpa/day]'
            vmin = None
            vmax = None
            cmap = 'Blues'
            filename = f'wap_tMean.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['itcz_width_tMean']:
            print(ds_itcz_tMean)
            if switch_test['scatter_org_tMean']:
                label_x = r'rome [km${^2}$]'
                label_y = r'obj_lat_variance [degrees]'
                filename = f'wap_rome_historical.png'
                x = ds_rome
                y = ds_itcz_tMean
                fig, ax = sL.plot_dsScatter(ds_x = x, ds_y = y, variable_list = list(x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                            xmin = None, xmax = None, ymin = None, ymax = None,
                            fig_given = False, fig = '', ax = '', 
                            color = 'k', models_highlight = [''], color_highlight = 'b', 
                            add_correlation = True, put_point = True)
                mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            

