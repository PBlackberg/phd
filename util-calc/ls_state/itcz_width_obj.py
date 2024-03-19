'''
# ----------------
#   itcz_obj
# ----------------
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



# --------------
#  obj location
# --------------
# ------------------------------------------------------------------------------- in latitude (obj) ----------------------------------------------------------------------------------------------------- #
def get_scene_obj_lats(scene, labels, dim):
    scene_obj_lats = []
    for i, label_i in enumerate(labels[0:-1]): 
        scene_i = scene.isin(label_i)
        scene_i = xr.where(scene_i > 0, 1, np.nan)
        scene_lat = dim.latm * scene_i
        scene_obj_lats.append(scene_lat.mean())
    return scene_obj_lats

def get_obj_lats(conv_obj, obj_id, dim):
    ''' This metric is based on the variance of object latitude position
        (object latitude position is calculated as the mean latitude position of gridboxes in object) '''
    obj_lats = []
    for timestep in np.arange(0, len(conv_obj.time.data)):
        if timestep % 365 == 0:
            print(f'\t Processing time: {str(conv_obj.time.data[timestep])[:-8]} ...')
        scene = conv_obj.isel(time = timestep)                                              # load scene
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').compute()             # load object index (can be masked)
        obj_lats.extend(get_scene_obj_lats(scene, labels_scene, dim))                       # all object latitudes
    return xr.DataArray(obj_lats, dims = ['obj'], coords = {'obj': np.arange(0, len(obj_lats))})

def get_obj_id_lats(conv_obj, obj_id, dim):
    ''' same as get_obj_lats, only separating daily lat positions in dimensions [day, obj].
        Masking can be applied to the result of this function '''
    obj_id_lats = xr.DataArray(np.nan, dims=obj_id.dims, coords=obj_id.coords)
    # obj_id_lats = xr.zeros_like(data_array)*np.nan
    # print(obj_id_lats)
    # exit()
    for timestep in np.arange(0, len(conv_obj.time.data)):
        if timestep % 365 == 0:
            print(f'\t Processing time: {str(conv_obj.time.data[timestep])[:-8]} ...')
        scene = conv_obj.isel(time = timestep)                                              # load scene
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').compute()             # load object index (can be masked)
        scene_lats = get_scene_obj_lats(scene, labels_scene, dim)
        obj_id_lats.isel(time=timestep, obj=slice(0, len(scene_lats)))[:] = scene_lats
    return obj_id_lats


# -------------------------------------------------------------------------------- in longitude (obj) ----------------------------------------------------------------------------------------------------- #
def get_scene_obj_lons(scene, labels, dim):
    scene_obj_lats = []
    for i, label_i in enumerate(labels[0:-1]): 
        scene_i = scene.isin(label_i)
        scene_i = xr.where(scene_i > 0, 1, np.nan)
        scene_lat = dim.latm * scene_i
        scene_obj_lats.append(scene_lat.mean())
    return scene_obj_lats

def get_obj_lons(conv_obj, obj_id, dim):
    ''' This metric is based on the variance of object latitude position
        (object latitude position is calculated as the mean latitude position of gridboxes in object) '''
    obj_lats = []
    for timestep in np.arange(0, len(conv_obj.time.data)):
        print(f'\t Processing {str(conv_obj.time.data[timestep])[:-8]} ...')
        scene = conv_obj.isel(time = timestep)                                              # load scene
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').compute()             # load object index (can be masked)
        obj_lats.extend(get_scene_obj_lats(scene, labels_scene, dim))                       # all object latitudes
    return xr.DataArray(obj_lats, dims = ['obj'], coords = {'obj': np.arange(0, len(obj_lats))})

def get_obj_id_lons(conv_obj, obj_id, dim):
    ''' same as get_obj_lats, only separating daily lat positions in dimensions [day, obj].
        Masking can be applied to the result of this function '''
    obj_id_lats = xr.DataArray(np.nan, dims=obj_id.dims, coords=obj_id.coords)
    # obj_id_lats = xr.zeros_like(data_array)*np.nan
    # print(obj_id_lats)
    # exit()
    for timestep in np.arange(0, len(conv_obj.time.data)):
        print(f'\t Processing {str(conv_obj.time.data[timestep])[:-8]} ...')
        scene = conv_obj.isel(time = timestep)                                              # load scene
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').compute()             # load object index (can be masked)
        scene_lats = get_scene_obj_lons(scene, labels_scene, dim)
        obj_id_lats.isel(time=timestep, obj=slice(0, len(scene_lats)))[:] = scene_lats
    return obj_id_lats



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
    import missing_data             as mD
    import dimensions_data          as dD
    import variable_calc            as vC
    import metric_data              as mDd
    import var_calc.conv_obj_var    as cO

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.line_plot       as lP
    import get_plot.scatter_time    as sT
    import get_plot.map_plot        as mP
    import get_plot.scatter_label   as sL

    switch_method = {
        'obj_lat':  True,
        'obj_lon':  False,
        }

    switch = {
        'test_sample':  False, 
        'fixed_area':   False,
        }

    switch_test = {
        'delete_previous_plots':        True,               # 1
        'scene':                        True,               # 2
        'tMean':                        True,               # 3
        'tMean_change':                 True,               # 4

        'obj_location':                 True,              # 5
            'percentiles':                      True,      # 6
            'freq_occur':                       True,      # 7
            'scatter_org_daily':                True,      # 8

            'scatter_obj_loc_var_org':          True,      # 9
            'scatter_obj_loc_var_org_change':   True,      # 10
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
    experiment = cD.experiments[0]
    
    for var_name in [k for k, v in switch_method.items() if v]:
        print(f'running {var_name}')
        ds_snapshot = xr.Dataset()
        ds_tMean, ds_tMean_change = xr.Dataset(), xr.Dataset()
        ds_percentile_value, ds_freq_occur, ds_daily_loc = xr.Dataset(), xr.Dataset(), xr.Dataset() 
        ds_obj_loc_var, ds_obj_loc_var_change = xr.Dataset(), xr.Dataset()
        ds_rome, ds_rome_change = xr.Dataset(), xr.Dataset()
        ds_tas_change = xr.Dataset()
        for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
            print(f'dataset: {dataset}')
            # ----------------------------------------------------------------------------------- Get data -------------------------------------------------------------------------------------------------- #
            conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
            obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
            dim = dD.dims_class(conv_obj)
            rome = mDd.load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = experiment, timescale = 'daily')
            ds_rome[dataset] = lP.pad_length_difference(rome) 
            if switch_test['tMean_change'] or switch_test['scatter_obj_loc_var_org_change']:
                conv_obj_warm, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
                obj_id_warm, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = 'ssp585', resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
                tas = mDd.load_metric(metric_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = experiment, timescale = 'monthly').mean(dim = 'time')   
                tas_warm = mDd.load_metric(metric_type = 'tas', metric_name = f'tas_sMean', dataset = dataset, experiment = 'ssp585', timescale = 'monthly').mean(dim = 'time')   
                ds_tas_change[dataset] = tas_warm - tas
                if switch_test['scatter_obj_loc_var_org_change']:
                    rome_warm = mDd.load_metric(metric_type = 'conv_org', metric_name = f'rome_{cD.conv_percentiles[0]}thprctile', dataset = dataset, experiment = 'ssp585', timescale = 'daily')
                    ds_rome_change[dataset] = (lP.pad_length_difference(rome_warm).mean(dim = 'time') - lP.pad_length_difference(rome).mean(dim = 'time')) / ds_tas_change[dataset]


            # print(f'conv_obj: \n {conv_obj}')
            # print(f'obj_id: \n {obj_id}')
            # print(f'conv_obj_warm: \n {conv_obj_warm}')
            # print(f'obj_id_warm: \n {obj_id_warm}')
            # print(f'tas: \n {tas}')
            # print(f'tas_warm: \n {tas_warm}')
            # print(f'rome: \n {rome}')
            # print(f'rome_warm: \n {rome_warm}')
            # exit()


            # ----------------------------------------------------------------------------------- Calculate -------------------------------------------------------------------------------------------------- #
            if switch_test['scene']:
                if var_name == 'obj_lat':
                    ds_snapshot[dataset] = xr.where(conv_obj.isel(time = 0) > 0, 1, 0) 

            if switch_test['tMean']:
                if var_name == 'obj_lat':
                    conv = xr.where(conv_obj > 0, 1, 0) 
                    ds_tMean[dataset] = conv.mean(dim = 'time') * 100
                    ds_tMean_change[dataset] =  ((xr.where(conv_obj_warm > 0, 1, 0).mean(dim = 'time') * 100) - (xr.where(conv_obj > 0, 1, 0).mean(dim = 'time')* 100)) /ds_tas_change[dataset] if switch_test['tMean_change'] else None

            if switch_test['obj_location']:
                if var_name == 'obj_lat':
                    obj_loc = get_obj_lats(conv_obj, obj_id, dim)
                    obj_loc_warm = get_obj_lats(conv_obj_warm, obj_id_warm, dim)    if switch_test['scatter_obj_loc_var_org_change'] else None

                if switch_test['percentiles']:
                    ds_percentile_value[dataset] = get_percentile_values(np.abs(obj_loc), dimensions = 'obj', percentiles = np.arange(1, 101, 1))
                    x_percentiles = ds_percentile_value['quantile'] * 100

                if switch_test['freq_occur']:
                    ds_freq_occur[dataset], bins_middle = calc_freq_occur(np.abs(obj_loc), ymin = 0, ymax = 30, nb_bins = 10)
                    x_freq_occur = bins_middle     

                if switch_test['scatter_org_daily']:
                    if var_name == 'obj_lat':
                        obj_id_daily = np.abs(get_obj_id_lats(conv_obj, obj_id, dim)) # distance from equator in degrees     
                        obj_id_daily = obj_id_daily.mean(dim='obj')
                    ds_daily_loc[dataset] = lP.pad_length_difference(obj_id_daily) 

                if switch_test['scatter_obj_loc_var_org'] or switch_test['scatter_obj_loc_var_org_change']:
                    if var_name == 'obj_lat':
                        # metric = calc_mae(obj_lat, reference_point = 0)
                        metric = calc_variance(obj_loc, reference_point = 0)
                        metric_warm = calc_variance(obj_loc_warm, reference_point = 0) if switch_test['scatter_obj_loc_var_org_change'] else None
                    ds_obj_loc_var[dataset] = metric
                    ds_obj_loc_var_change[dataset] =  (metric_warm - metric) / ds_tas_change[dataset] if switch_test['scatter_obj_loc_var_org_change'] else None

        # print(ds_obj_loc_var)
        # exit()
        # ------------------------------------------------------------------------------------------ Plot -------------------------------------------------------------------------------------------------- #
        if switch_test['scene'] and var_name == 'obj_lat':
            if var_name == 'obj_lat':
                ds = ds_snapshot
                label = 'convection [0,1]'
                vmin = None
                vmax = None
                cmap = 'Blues'
                filename = f'conv.png'
                fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
                mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
        
        if switch_test['tMean'] and var_name == 'obj_lat':
            ds = ds_tMean
            label = 'Object frequency of occurence [%]'
            vmin = 0
            vmax = 30
            cmap = 'Blues'
            filename = f'conv_tMean.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['tMean_change'] and var_name == 'obj_lat':
            ds = ds_tMean_change
            label = 'Object frequency of occurence [% / K]'
            vmin = -10
            vmax = 10
            cmap = 'RdBu_r'
            filename = f'conv_tMean_change.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['percentiles']:
            ds = ds_percentile_value
            label_y = 'percentile value'
            label_x = 'percentile'
            filename = f'{var_name}_value_distribution_percentiles.png'
            ymin_list, ymax_list = [], []
            for variable in ds.data_vars:
                ymin_list.append(ds[variable].min())
                ymax_list.append(ds[variable].max())
            ymin = min(ymin_list)
            ymax = max(ymax_list)
            fig, axes = lP.plot_dsLine(ds, x = x_percentiles, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x, label_y = label_y, colors = ['k', 'r'], 
                        xmin = None, xmax = None, ymin = ymin, ymax = ymax,
                        fig_given = False, one_ax = True, fig = '', axes = '',
                        distinct_colors = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['freq_occur']:
            ds = ds_freq_occur
            label_y = 'Freq. occur [%]'
            label_x = 'latitude [degree]' if var_name == 'obj_lat' else 'longitude [degree]'
            filename = f'{var_name}_freq_occur.png'
            ymin_list, ymax_list = [], []
            for variable in ds.data_vars:
                ymin_list.append(ds[variable].min())
                ymax_list.append(ds[variable].max())
            ymin = min(ymin_list)
            ymax = max(ymax_list)
            print(x_freq_occur)
            fig, axes = lP.plot_dsLine(ds, x = x_freq_occur, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x, label_y = label_y, colors = ['k', 'r'], 
                        xmin = None, xmax = None, ymin = ymin, ymax = ymax,
                        fig_given = False, one_ax = False, fig = '', axes = '',
                        distinct_colors = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['scatter_org_daily']:
            ds_x = ds_rome
            ds_y = ds_daily_loc
            label_cmap = 'days' #'months [nb]'
            vmin = None #0 
            vmax = None #10
            label_x = r'rome [km${^2}$]'    
            label_y = r'latitude [degrees]'    if var_name == 'obj_lat'   else r'longitude [degrees]'        
            filename = f'{var_name}_scatter.png'
            fig, axes = sT.plot_dsScatter(ds_x = ds_x, ds_y = ds_y, variable_list = list(ds_x.data_vars.keys()), 
                                        fig_title = filename, label_x = label_x, label_y = label_y, label_cmap = label_cmap,
                                        colors_scatter = ['k']*len(ds_x.data_vars), colors_slope = ['k']*len(ds_x.data_vars), cmap = 'Blues',
                                        xmin = None, xmax = None, ymin = None, ymax = None, vmin = vmin, vmax = vmax,
                                        density_map = False, models_highlight = ['a', 'b'],
                                        fig_given = False, ax = '', individual_cmap = False)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

        if switch_test['scatter_obj_loc_var_org']:
            label_x = r'rome [km${^2}$]'
            label_y = r'obj_lat_variance [degrees]' if var_name == 'obj_lat'   else r'obj_lon_variance [degrees]'     
            filename = f'{var_name}_rome_historical.png'
            ds_x = ds_rome.mean(dim='time')
            ds_y = ds_obj_loc_var
            fig, ax = sL.plot_dsScatter(ds_x = ds_x, ds_y = ds_y, variable_list = list(ds_x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            
        if switch_test['scatter_obj_loc_var_org_change']:
            label_x = r'rome [km${^2}$K${^-1}$]'
            label_y = r'obj_lat_variance [degrees]' if var_name == 'obj_lat'   else r'obj_lon_variance [degrees]'     
            filename = f'{var_name}_rome_change_with_warming.png'
            ds_x = ds_rome_change
            ds_y = ds_obj_loc_var_change
            fig, ax = sL.plot_dsScatter(ds_x = ds_x, ds_y = ds_y, variable_list = list(ds_x.data_vars.keys()), fig_title = filename, x_label = label_x, y_label = label_y, 
                        xmin = None, xmax = None, ymin = None, ymax = None,
                        fig_given = False, fig = '', ax = '', 
                        color = 'k', models_highlight = [''], color_highlight = 'b', 
                        add_correlation = True, put_point = True)
            mP.show_plot(fig, show_type = 'save_cwd', filename = filename)


