'''
# ------------------------
#  ITCZ width calculation
# ------------------------
Based on;
- time-mean vertical pressure velocity at 500 hpa
- variance in object location from the equator (precipitation maxima)
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np



# -------------
#  itcz width
# -------------
# -------------------------------------------------------------------------- Based on vertical pressure velocity (wap) ----------------------------------------------------------------------------------------------------- #
def wap_itcz_width(da):
    da = da.mean(dim = ['lon'])
    itcz = xr.where(da < 0, 1, np.nan).compute()        
    itcz_lats = itcz * da['lat']                                # lats in ascent
    return itcz_lats.max(dim='lat') - itcz_lats.min(dim='lat')

def itcz_width_tMean(da):
    alist = da.mean(dim = ['time', 'lon']).compute()
    itcz_lats = alist.where(alist < 0, drop = True)['lat']
    return itcz_lats.max() - itcz_lats.min()

def get_fraction_descent(da, dims):
    da = da.mean(dim = 'time')
    da = ((xr.where(da > 0, 1, 0) * dims.aream).sum() / dims.aream.sum())*100   # fraction of descent
    return da                              


# -------------------------------------------------------------------------------- Based on object location (obj) ----------------------------------------------------------------------------------------------------- #
def get_obj_lat(scene, labels, dim):
    obj_lats = []
    for i, label_i in enumerate(labels[0:-1]): 
        scene_i = scene.isin(label_i)
        scene_i = xr.where(scene_i > 0, 1, np.nan)
        scene_lat = dim.latm * scene_i
        obj_lats.append(scene_lat.mean())
    return obj_lats

def get_obj_itcz_width(conv_obj, obj_id, dim):
    ''' This metric is based on the variance of object latitude position
        (object latitude position is calculated as the mean latitude position of gridboxes in object) '''
    obj_lat = []
    for timestep in np.arange(0,2): #len(conv_obj.time.data)
        print(f'\t Processing {str(conv_obj.time.data[timestep])[:-8]} ...')
        scene = conv_obj.isel(time = timestep)                                              # load scene
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').compute()             # load object index (can be masked)
        obj_lat.extend(get_obj_lat(scene, labels_scene, dim))
    return obj_lat



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

    switch_method = {
        'wap':  True,
        'obj':  False,
        }
    
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
        'scene':                        True,
        'tMean':                        True,
        'itcz_width':                   True,
        'plot_change':                  False,
        'variance':                     False,
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None

    experiment = cD.experiments[0]
    
    for var_name in [k for k, v in switch_method.items() if v]:
        print(f'running {var_name} method')

        ds_wap = xr.Dataset()
        ds_wap_tMean, ds_wap_tMean_change = xr.Dataset(), xr.Dataset()
        ds_wap_itcz, ds_wap_itcz_change = xr.Dataset(), xr.Dataset()
        ds_wap_itcz_sMean = xr.Dataset()

        ds_obj = xr.Dataset()
        ds_obj_tMean, ds_obj_tMean_change = xr.Dataset(), xr.Dataset()
        ds_obj_itcz, ds_obj_itcz_change = xr.Dataset(), xr.Dataset()
        ds_obj_itcz_sMean = xr.Dataset()

        ds_rome, ds_rome_change = xr.Dataset(), xr.Dataset()
        ds_tas_change = xr.Dataset()
        for dataset in mD.run_dataset_only(var = 'pe', datasets = cD.datasets):
            print(f'dataset: {dataset}')
            # ----------------------------------------------------------------------------------- Get data -------------------------------------------------------------------------------------------------- #
            wap, region = vC.get_variable(switch_var = {'wap': True}, switch = switch_wap, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = False, re_process = False)
            wap_warm, region = vC.get_variable(switch_var = {'wap': True}, switch = switch_wap, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = False, re_process = False)
            

            conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch_obj, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
            obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch_obj, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
            # conv_obj_warm, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch_obj, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
            # obj_id_warm, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch_obj, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'daily', from_folder = True, re_process = False)
            dim = dD.dims_class(conv_obj)

            # print(f'wap: \n {wap}')
            # print(f'conv_obj: \n {conv_obj}')
            # print(f'obj_id: \n {obj_id}')
            # exit()

            # ----------------------------------------------------------------------------------- Calculate -------------------------------------------------------------------------------------------------- #
            if switch_test['scene']:
                if var_name == 'wap':
                    ds_wap[dataset] = wap.isel(time = 0) 
                if var_name == 'obj':
                    ds_obj[dataset] = xr.where(conv_obj.isel(time = 0) > 0, 1, 0) 

            if switch_test['tMean']:
                if var_name == 'wap':
                    ds_wap_tMean[dataset] = wap.mean(dim = 'time') 
                if var_name == 'obj':
                    conv = xr.where(conv_obj > 0, 1, 0) 
                    ds_obj[dataset] = conv.mean(dim = 'time')

            if switch_test['itcz_width']:
                if var_name == 'wap':
                    ds_wap_itcz[dataset] = itcz_width_tMean(wap)
                if var_name == 'obj':
                    ds_obj_itcz[dataset] = get_obj_itcz_width(conv_obj, obj_id, dim)



        # x = da_mean_lat.values
        # y = da_mean_lat['lat'].values
        # fig = plt.figure(figsize=(8, 6))
        # plt.plot(x, y, 'k')
        # plt.xlabel('time-lon mean wap')
        # plt.ylabel('lat')
        # plt.axvline(x=0, color='k', linestyle='--')
        # plt.axhline(y=0, color='k', linestyle='--')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_tMean_lonMean')
        
        
            
