''' 
# ------------------------
#      ROME calc
# ------------------------
ROME - Radar Organization MEtric 
Evaluated from unique object pair weight (q_i): 
q_i(A_a, A_b, A_d) = A_a + min(1, A_b / A_d) * A_b
where
object  - contigous convective regions
A_a     - larger area of pair
A_b     - smaller area of pair
A_d     - squared shortest distance between pair boundaries (km)
and
ROME = mean(q_i)

to use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-calc')
import conv_org.rome_calc as rC
aList = calc_rome(conv_obj, obj_id, dim)
'''



# ------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import dask
from distributed import Client, LocalCluster
import itertools
import glob
from pathlib import Path
# from functools import wraps
# from datetime import datetime


# --------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_calc as vC

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF

import time

def timing_decorator(func):
    """
    A decorator to measure the execution time of a function.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record the end time
        time_taken = end_time - start_time  # Calculate the time taken
        print(f"Function {func.__name__} took {time_taken:.4f} seconds to execute.")
        return result
    return wrapper



# ----------
#   ROME
# ----------
# ---------------------------------------------------------------------------------- Calculate weights --------------------------------------------------------------------------------------------------- #
def calc_q_weight(scene, labels, distance_matrix, dim):
    ''' Create unique pair weight: q = A_a + min(1, A_b / A_d) * A_b 
        Where
        A_a     - larger area of pair
        A_b     - smaller area of pair
        A_d     - squared shortest distance between pair boundaries (km)
    '''
    q_pairs_day = []
    for i, label_i in enumerate(labels[0:-1]): 
        try:
            scene_i = scene.isin(label_i)
            scene_i = xr.where(scene_i > 0, 1, 0)
            I, J = zip(*np.argwhere(scene_i.data)) 
            I, J = list(I), list(J)
            I, J = np.array(I), np.array(J)
            # print(I)
            # print(J)
            n_list = I * len(distance_matrix['lon']) + J
            # print(n_list)
            distance_from_obj_i = distance_matrix.sel(gridbox = n_list).min(dim = 'gridbox')
            # exit()
            obj_i_area = (dim.aream * scene_i).sum()
            for _, label_j in enumerate(labels[i+1:]):
                scene_j = scene.isin(label_j)
                scene_j = xr.where(scene_j > 0, 1, np.nan)
                A_d = (distance_from_obj_i * scene_j).min()**2
                obj_j_area = (dim.aream * scene_j).sum()    
                A_a, A_b = sorted([obj_i_area, obj_j_area], reverse=True)
                q_weight = A_a + np.minimum(A_b, (A_b / A_d) * A_b)
                q_pairs_day.append(q_weight)
        except Exception as e:
            print(f"Error occurred at iteration with label_i = {label_i}, i = {i}")
            raise e 
    return np.mean(np.array(q_pairs_day))


# ---------------------------------------------------------------------------------- Calculate daily values --------------------------------------------------------------------------------------------------- #
def calc_rome_day(scene, labels_scene, distance_matrix, dim, day):
    # print(f'processing day: {day}')
    if len(labels_scene) == 1:
        rome_day = ((scene.isin(labels_scene)>0)*1 * dim.aream).sum()
        return rome_day
    else:
        rome_day = calc_q_weight(scene, labels_scene, distance_matrix, dim)
        return xr.DataArray([rome_day], dims = ['time'], coords = {'time': [scene['time'].data]})


# ------------------
#  parallelize calc
# ------------------
# -------------------------------------------------------------------------------- manage timecoordinate --------------------------------------------------------------------------------------------------- #
def time_to_index(da):
    ''' time coordinates cannot be cftime when scattering data to workers '''
    return da.assign_coords(time=np.arange(len(da['time'])))

def time_index_dict(time_coords):
    ''' Divide indices into the years of the time coordinates '''
    month_list = time_coords.dt.month
    unique_months = np.unique(month_list)
    return {month: np.where(month_list == month)[0] for month in unique_months}
    

# ------------------------------------------------------------------------------ distribute data to workers --------------------------------------------------------------------------------------------------- #
def scatter_data(data_object, client):
    return client.scatter(data_object, direct=True, broadcast=True)     


# -------------------------------------------------------------------------------- set up parallelization --------------------------------------------------------------------------------------------------- #
@dask.delayed
def calc_rome_month(data_objects, month, indices):
    print(f'processing month: {month}')
    conv_obj_year, obj_id_year, distance_matrix, dim = data_objects
    conv_obj_month = conv_obj_year.isel(time = indices)
    obj_id_month = obj_id_year.isel(time = indices).dropna(dim='obj')
    return [calc_rome_day(conv_obj_month.isel(time = day), obj_id_month.isel(time = day), distance_matrix, dim, day) for day in range(len(conv_obj_month.time))]

@timing_decorator
def calc_rome_year(conv_obj, obj_id, distance_matrix, dim, year, path_year):
    print(f'processing year: {year} ..')
    conv_obj_year = conv_obj.sel(time=f'{year}')
    time_coords = conv_obj_year['time']
    conv_obj_year, obj_id_year = [time_to_index(da) for da in [conv_obj_year, obj_id.sel(time=f'{year}')]] # convert time to index
    year_indices_dict = time_index_dict(time_coords)
    data_objects = [conv_obj_year, obj_id_year, distance_matrix, dim]
    data_object_copies = [scatter_data(data_object, client) for data_object in data_objects] # give data to workers # dim
    
    # futures = [calc_rome_month(data_object_copies, month, indices) for month, indices in year_indices_dict.items()] # list of futures (task graphs)
    # results = dask.compute(*futures)                                                                        # each future becomes a list
    # results = list(itertools.chain.from_iterable(results))                                                  # flatten the list of lists into one list
    
    chunks = [dict(itertools.islice(year_indices_dict.items(), i, i+4)) for i in range(0, len(year_indices_dict), 4)]
    all_results = []  # This will store the results of all chunks
    for chunk in chunks:
        futures = [calc_rome_month(data_object_copies, month, indices) for month, indices in chunk.items()]
        results = dask.compute(*futures)  # Execute the futures
        results = list(itertools.chain.from_iterable(results))  # Flatten the list of lists
        all_results.extend(results)  # Accumulate the results
    rome_year = xr.concat(all_results, dim='time')
    # print(rome_year)
    xr.Dataset({'rome': rome_year}).to_netcdf(path_year, mode="w")

def calc_rome(conv_obj, obj_id, distance_matrix, dim, folder, metric_name):
    print('calc_rome started')
    path_years = []
    for year in np.unique(conv_obj['time'].dt.year): 
        path_year = f'{folder}/{metric_name}_{year}.nc'
        if os.path.exists(path_year):
            path_years.append(path_year)
            print(f"results for year {year} already exists. Skipping...")
        else:
            calc_rome_year(conv_obj, obj_id, distance_matrix, dim, year, path_year)
            path_years.append(path_year)
            print(f'ROME for year {year} saved in temp folder.')
    return path_years


# ------------------------
#  Call function and save
# ------------------------
def get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = 'dataset_experiment'):
    metric_name = 'rome'
    folder = f'{sF.folder_scratch}/temp_calc/{metric_name}/{folder_temp}'
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
    path_years = calc_rome(conv_obj, obj_id, distance_matrix, dim, folder, metric_name)     # create files for individual years
    # print(path_years)
    # ds = xr.open_dataset(path_years[0])
    # print(ds)
    # exit()
    ds = xr.open_mfdataset(path_years, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True) 
    ds = ds.assign_coords(time=conv_obj['time'])
    ds = ds.load()
    print(ds['rome'])
    exit()
    for path_year in path_years:                                                            # remove temp_calc files after files have been combined
        os.remove(path_year)
    return ds['rome']



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('rome test starting')
    def create_client():
        cluster = LocalCluster(n_workers = 4, threads_per_worker = 2, memory_limit = '4GB')
        client = Client(cluster)
        print(client)
        print(f"Dask Dashboard is available at {client.dashboard_link}")
        import webbrowser
        webbrowser.open(f'{client.dashboard_link}') 
        # print('for dashboard, open: http://127.0.0.1:8787/status') # this is the link that is given when opening the browser from the login node
        return client
    client = create_client()

    import os
    import sys
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data             as mD
    import dimensions_data          as dD
    import var_calc.conv_obj_var    as cO

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.line_plot   as lP
    import get_plot.map_plot    as mP

    switch = {
        'test_sample':  False, 
        'fixed_area':   False
        }

    switch_test = {
        'delete_previous_plots':    True,
        'rome':                     True,
        'plot_objects':             False,
        'rome_subset':              False,
        'plot_objects_subset':      False,
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
    
    experiment = cD.experiments[0]
    distance_matrix = xr.open_dataset(f'{sF.folder_scratch}/sample_data/distance_matrix/distance_matrix_{int(360/cD.x_res)}x{int(180/cD.y_res)}.nc')['distance_matrix']
    distance_matrix.load()
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

        # conv_obj = conv_obj.sel(time = slice('1970-01-01', '1970-08'))
        # obj_id = obj_id.sel(time = slice('1970-01-01', '1970-08'))

        dim = dD.dims_class(conv_obj)

        # print(f'distance_matrix: \n {distance_matrix}')
        # print(f'conv_obj: \n {conv_obj}')
        # print(f'obj_id: \n {obj_id}')
        # print(f'timeaxis: \n {conv_obj.time}')
        # exit()


        # --------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
        if switch_test['rome']:
            rome = get_rome(conv_obj, obj_id, distance_matrix, dim, f'{dataset}_{experiment}')
            ds_rome[dataset] = lP.pad_length_difference(rome)
            # print(rome)

        if switch_test['plot_objects']:
            ds_obj[f'{dataset}_rome_{str(format(ds_rome[dataset].isel(time=0).data, ".2e"))}'] = xr.where(conv_obj.isel(time = 0) > 0, 1, 0)

        if switch_test['rome_subset']:
            metric_id_mask = cO.create_mock_metric_mask(obj_id)
            conv_obj_subset = cO.get_obj_subset(conv_obj, obj_id, metric_id_mask)
            print(f'metric id mask: \n {metric_id_mask}')
            print(f'conv_obj_subset: \n {conv_obj_subset}')
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
        print('ROME from subset results: \n')
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

    client.close()




    # path_pattern = f'{folder}/{metric_name}_*.nc'
    # paths = glob.glob(path_pattern)