''' 
# ------------------------
#      ROME calc
# ------------------------
ROME - Radar Organization MEtric 
Evaluated from unique object pair weight: 
q(A_a, A_b, A_d) = A_a + min(1, A_b / A_d) * A_b
where
object  - contigous convective regions
A_a     - larger area of pair
A_b     - smaller area of pair
A_d     - squared shortest distance between pair boundaries (km)
ROME = mean(q)

to use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-calc')
import conv_org.rome_calc as rC
aList = calc_rome(conv_obj, obj_id, dim)
'''
# import warnings
# warnings.filterwarnings('error')


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
# from distributed import Client
# from dask import delayed, compute


# ---------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_calc            as vC



# ----------
#   ROME
# ----------
def calc_q_weight(scene, labels, distance_matrix, dim):
    ''' Create unique pair weight: q = A_a + min(1, A_b / A_d) * A_b 
        Where
        A_a     - larger area of pair
        A_b     - smaller area of pair
        A_d     - squared shortest distance between pair boundaries (km)
    '''
    print('weight calc started')
    q_pairs_day = []
    for i, label_i in enumerate(labels[0:-1]): 
        scene_i = scene.isin(label_i)
        scene_i = xr.where(scene_i > 0, 1, 0)
        I, J = zip(*np.argwhere(scene_i.data)) 
        I, J = list(I), list(J)
        I, J = np.array(I), np.array(J)
        n_list = I * len(distance_matrix['lon']) + J
        distance_from_obj_i = distance_matrix.sel(gridbox = n_list).min(dim = 'gridbox')
        obj_i_area = (dim.aream * scene_i).sum()
        for _, label_j in enumerate(labels[i+1:]):
            scene_j = scene.isin(label_j)
            scene_j = xr.where(scene_j > 0, 1, np.nan)
            A_d = (distance_from_obj_i * scene_j).min()**2
            obj_j_area = (dim.aream * scene_j).sum()    
            A_a, A_b = sorted([obj_i_area, obj_j_area], reverse=True)
            q_weight = A_a + np.minimum(A_b, (A_b / A_d) * A_b)
            q_pairs_day.append(q_weight)
    return np.mean(np.array(q_pairs_day))

def calc_rome_day(scene, labels_scene, distance_matrix, dim, day):
    print(f'processing day {day}')
    if len(labels_scene) == 1:
        return ((scene.isin(labels_scene)>0)*1 * dim.aream).sum()
    else:
        return calc_q_weight(scene, labels_scene, distance_matrix, dim)

def get_rome(conv_obj, obj_id, distance_matrix, dim):
    print('rome_calc started')
    rome_list = [calc_rome_day(conv_obj.isel(time=timestep), obj_id.isel(time=timestep).dropna(dim='obj'), distance_matrix, dim, timestep) for timestep in range(len(conv_obj.time))]
    return xr.DataArray(rome_list, dims = ['time'], coords = {'time': conv_obj.time.data})



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('rome test starting')
    import os
    import sys
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  

    sys.path.insert(0, f'{os.getcwd()}/util-files')
    import save_folders as sF

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data             as mD
    import dimensions_data          as dD
    import var_calc.conv_obj_var    as cO

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.line_plot   as lP
    import get_plot.map_plot    as mP

    switch = {
        'test_sample': False, 
        'fixed_area': False
        }

    switch_test = {
        'delete_previous_plots':    False,
        'rome':                     True,
        'plot_objects':             False,
        'rome_subset':              False,
        'plot_objects_subset':      False,
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
    
    experiment = cD.experiments[0]
    distance_matrix = xr.open_dataset(f'{sF.folder_scratch}/sample_data/distance_matrix/distance_matrix_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc')['distance_matrix']
    # print(distance_matrix)
    # print(format_bytes(distance_matrix.nbytes))
    # exit()

    ds_rome = xr.Dataset()
    ds_obj = xr.Dataset()
    ds_rome_subset = xr.Dataset()
    ds_obj_subset = xr.Dataset()
    for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
        print(f'dataset: {dataset}')
        # -------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
        obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
        conv_obj.load()
        obj_id.load()

        dim = dD.dims_class(conv_obj)

        print(f'distance_matrix: \n {distance_matrix}')
        print(f'conv_obj: \n {conv_obj}')
        print(f'obj_id: \n {obj_id}')
        # exit()


        # --------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
        if switch_test['rome']:
            rome = get_rome(conv_obj, obj_id, distance_matrix, dim)
            ds_rome[dataset] = lP.pad_length_difference(rome)
            # print(rome)

        if switch_test['plot_objects']:
            # ds_obj[f'{dataset}_rome_{str(format(ds_rome[dataset].isel(time=0).data, ".2e"))}'] = xr.where(conv_obj.isel(time = 0) > 0, 1, 0)
            ds_obj[f'{dataset}'] = xr.where(conv_obj.isel(time = 0) > 0, 1, 0)

        if switch_test['rome_subset']:
            metric_id_mask = cO.create_mock_metric_mask(obj_id)
            conv_obj_subset = cO.get_obj_subset(conv_obj, obj_id, metric_id_mask)
            # print(f'metric id mask: \n {metric_id_mask}')
            # print(f'metric id mask: \n {conv_obj_subset}')
            # print(f'unique subset: \n {np.unique(conv_obj_subset)}')
            obj_id_masked = obj_id * metric_id_mask
            rome = get_rome(conv_obj, obj_id_masked, distance_matrix, dim)
            ds_rome_subset[dataset] = lP.pad_length_difference(rome)

        if switch_test['plot_objects_subset']:
            timestep = 0        
            labels_scene = obj_id_masked.isel(time = timestep).dropna(dim='obj').values
            scene = conv_obj.isel(time = timestep)
            scene_subset = scene.isin(labels_scene)
            scene_subset = xr.where(scene_subset > 0, 1, 0)
            ds_obj_subset[f'{dataset}_rome_{str(format(ds_rome_subset[dataset].isel(time=0).data, ".2e"))}'] = scene_subset


    # ------------------------------------------------------------------------------------------- Plot --------------------------------------------------------------------------------------------------- #
    if switch_test['rome']:
        print('ROME results: \n')
        [print(f'{dataset}: {ds_rome[dataset].data}') for dataset in ds_rome.data_vars]
            
    if switch_test['plot_objects']:
        ds = ds_obj
        label = 'convection [0,1]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'conv_rome.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['rome_subset']:
        print('ROME from subset results:')
        [print(f'{dataset}: {ds_rome_subset[dataset].data}') for dataset in ds_rome_subset.data_vars]

    if switch_test['plot_objects_subset']:
        ds = ds_obj_subset
        label = 'convection [0,1]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'conv_subset_rome.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)



