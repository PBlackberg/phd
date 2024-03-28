''' 
# ------------------------
#      ROME calc
# ------------------------
Calculates ROME - Radar Organization MEtric 
Function:
    rome = get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = f'{dataset}_{experiment}')

Input:
    conv_obj        - integer matrix indifying objects  dim: (time, lat, lon)       get from: util-data/var_calc/conv_obj_var.py
    obj_id          - included objects                  dim: (time, obj)            get from: util-data/var_calc/conv_obj_var.py
    distance_matrix - distance calculator               dim: (n_gridbox, lat, lon)  get from: util-data/var_calc/distance_matrix.py
    dim             - dimensions (object)                                           get from: util-data/dimensions_data.py
    folder_temp     - folder in scratch to save partial calc

Output:
    rome:           - list                              dim: (time)
'''


# ------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import dask
from distributed import Client, LocalCluster, get_client
import itertools
from pathlib import Path
import time


# --------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_calc as vC

sys.path.insert(0, f'{os.getcwd()}/util-files')
import save_folders as sF



# ---------------
#  General funcs
# ---------------
def timing_decorator(func):
    ''' execution time of a function '''
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record the start time
        result = func(*args, **kwargs)  # Call the decorated function
        end_time = time.time()  # Record the end time
        time_taken = (end_time - start_time) / 60  # Calculate the time taken in minutes
        print(f"Function {func.__name__} took {time_taken:.4f} minutes to execute.")
        return result
    return wrapper

def ensure_client(func):
    ''' If a function in this script that uses dask is called from a different script, the client can be created or 
        temporarily be scaled and then reverted back to intial state '''
    def wrapper(*args, **kwargs):
        client_exists = True
        try:
            preferred_workers = 12
            preferred_threads = 2
            preferred_mem = '4GB'
            client = get_client()
            initial_workers = len(client.cluster.workers)
            print(f'initial client has {initial_workers} workers')
            if initial_workers == preferred_workers:  # scale client to set number of workers
                print(f'initial client has suitible number of workers')     
            else:
                print(f'Scaling number of client workers temporarily..')
                client.cluster.scale(preferred_workers)
                print(client)
        except ValueError:
            print(f'No initial client \n creating client..')
            client_exists = False
            cluster = LocalCluster(n_workers=preferred_workers, threads_per_worker=preferred_threads, memory_limit=preferred_mem)
            client = Client(cluster)
            print(client)
            print(f"Dask Dashboard is available at {client.dashboard_link}")
            import webbrowser
            webbrowser.open(f'{client.dashboard_link}') 
            # # print('for dashboard, open: http://127.0.0.1:8787/status') # this is the link that is given when opening the browser from the login node
        result = func(*args, **kwargs)
        if client_exists and initial_workers != preferred_workers:  # scale client back to what it was
            print(f'Changing back number of workers to initial state..')
            client.cluster.scale(initial_workers)
            print(client)
        if not client_exists:
            print(f'closing client, as no initial client was given - does not close for the moment')
            # client.close()
        return result
    return wrapper



# ------------
#   ROME calc
# ------------
# ---------------------------------------------------------------------------------- Calculate weights --------------------------------------------------------------------------------------------------- #
def calc_q_weights(scene, labels, distance_matrix, dim):
    ''' Create unique pair weight: 
        q = A_a + min(1, A_b / A_d) * A_b 
        Where
            A_a - larger area of pair
            A_b - smaller area of pair
            A_d - squared shortest distance between pair boundaries (km) '''
    q_pairs_day = []
    for i, label_i in enumerate(labels[0:-1]): 
        scene_i = scene.isin(label_i)
        scene_i = xr.where(scene_i > 0, 1, 0)
        I, J = zip(*np.argwhere(scene_i.data)) 
        I, J = list(I), list(J)
        I, J = np.array(I), np.array(J)
        n_list = I * len(dim.lon) + J
        distance_from_obj_i = distance_matrix.isel(gridbox = n_list).min(dim = 'gridbox')
        obj_i_area = (dim.aream * scene_i).sum()
        for _, label_j in enumerate(labels[i+1:]):
            scene_j = scene.isin(label_j)
            scene_j = xr.where(scene_j > 0, 1, np.nan)
            A_d = (distance_from_obj_i * scene_j).min()**2
            obj_j_area = (dim.aream * scene_j).sum()    
            A_a, A_b = sorted([obj_i_area, obj_j_area], reverse=True)
            q_weight = A_a + np.minimum(1, A_b / A_d) * A_b
            q_pairs_day.append(q_weight)
    return np.array(q_pairs_day)


# ---------------------------------------------------------------------------------- Calculate daily values --------------------------------------------------------------------------------------------------- #
def calc_rome_day(scene, labels_scene, distance_matrix, dim, day):
    ''' Calculate ROME:
        ROME = sum(A_i)     if n = 1
        ROME = mean(q_i)    if n > 1
        Where
            n - Number of objects in scene 
            A_i - Area of object
            q_i - weight assigned to unique object pairs '''
    # print(f'processing day: {day}')
    if len(labels_scene) == 1:
        rome_day = ((scene.isin(labels_scene)>0)*1 * dim.aream).sum()
        return rome_day
    else:
        rome_day = calc_q_weights(scene, labels_scene, distance_matrix, dim)
        return xr.DataArray([np.mean(rome_day)], dims = ['time'], coords = {'time': [scene['time'].data]})


# ------------------
#  parallelize calc
# ------------------
# -------------------------------------------------------------------------------- manage timecoordinate --------------------------------------------------------------------------------------------------- #
def convert_time_to_index(da):
    ''' time coordinates cannot be cftime when scattering data to workers '''
    return da.assign_coords(time=np.arange(len(da['time'])))

def time_index_dict(time_coords, time_period = ''):
    ''' Divide indices into the years of the time coordinates '''
    if time_period == 'year':
        year_list = time_coords.dt.year
        unique_years = np.unique(year_list)
        index_dict = {year: np.where(year_list == year)[0] for year in unique_years}   # find the indices for the year
    if time_period == 'month':
        month_list = time_coords.dt.month
        unique_months = np.unique(month_list)
        index_dict = {month: np.where(month_list == month)[0] for month in unique_months}
    return index_dict
    

# ------------------------------------------------------------------------------ distribute data to workers --------------------------------------------------------------------------------------------------- #
def scatter_data(data_object, client):
    return client.scatter(data_object, direct=True, broadcast=True)     


# -------------------------------------------------------------------------------- set up parallelization --------------------------------------------------------------------------------------------------- #
@dask.delayed
def calc_rome_month(data_objects, month, indices):
    ''' Calculate ROME for a month - this function is executed in parallel '''
    print(f'processing month: {month}')
    conv_obj_year, obj_id_year, distance_matrix, dim = data_objects
    conv_obj_month = conv_obj_year.isel(time = indices)
    obj_id_month = obj_id_year.isel(time = indices).dropna(dim='obj')
    return [calc_rome_day(conv_obj_month.isel(time = day), obj_id_month.isel(time = day), distance_matrix, dim, day) for day in range(len(conv_obj_month.time))]

def calc_rome_year(conv_obj, obj_id, distance_matrix, dim, year, path_year, year_indices_dict):
    ''' Calculate ROME for a year '''
    print(f'processing year: {year} ..')
    conv_obj_year = conv_obj.sel(time=f'{year}')
    time_coords = conv_obj_year['time']
    conv_obj_year, obj_id_year = [convert_time_to_index(da) for da in [conv_obj_year, obj_id.sel(time=f'{year}')]] # convert time to index
    month_indices_dict = time_index_dict(time_coords, 'month')
    data_objects = [conv_obj_year, obj_id_year, distance_matrix, dim]
    client = get_client()
    data_object_copies = [scatter_data(data_object, client) for data_object in data_objects] # give data to workers    
    chunk_size = len(client.cluster.workers) # Chunk size (number of months) matching number of workers, each worker works on one month independently
    chunks = [dict(itertools.islice(month_indices_dict.items(), i, i + chunk_size)) for i in range(0, len(month_indices_dict), chunk_size)]
    all_results = []
    for chunk in chunks:
        futures = [calc_rome_month(data_object_copies, month, indices) for month, indices in chunk.items()] # task graphs (futures) to be executes in parallel
        results = dask.compute(*futures)                        # ask workers to execute list of futures
        results = list(itertools.chain.from_iterable(results))  # each future in the list becomes a list itself, so flatten the list of lists
        all_results.extend(results)                             # accumulate the results from the chunks
    rome_year = xr.concat(all_results, dim='time')
    rome_year = rome_year.assign_coords(time = year_indices_dict[year])
    # print(rome_year)
    xr.Dataset({'rome': rome_year}).to_netcdf(path_year, mode="w")

def calc_rome(conv_obj, obj_id, distance_matrix, dim, folder, metric_name):
    ''' Create paths representing where the temp_data is stored'''
    print('calc_rome started')
    year_indices_dict = time_index_dict(conv_obj['time'], 'year') # get the index for the days in the years 
    path_years = []
    for year in np.unique(conv_obj['time'].dt.year): 
        path_year = f'{folder}/{metric_name}_{year}.nc'
        if os.path.exists(path_year):
            path_years.append(path_year)
            print(f"results for year {year} already exists. Skipping...")
        else:
            calc_rome_year(conv_obj, obj_id, distance_matrix, dim, year, path_year, year_indices_dict)
            path_years.append(path_year)
            print(f'ROME for year {year} saved in temp folder.')
    return path_years


# ------------------------
#  Call function and save
# ------------------------
@ensure_client
@timing_decorator
def get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = 'dataset_experiment'):
    ''' Concatenate the results from the yearly calculations '''
    metric_name = 'rome'
    folder = f'{sF.folder_scratch}/temp_calc/{metric_name}/{folder_temp}'
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)
    path_years = calc_rome(conv_obj, obj_id, distance_matrix, dim, folder, metric_name)     # create files for individual years (path_years is a list of file paths)
    # print(path_years)
    # ds = xr.open_dataset(path_years[0])
    # print(ds)
    # exit()
    ds = xr.open_mfdataset(path_years, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True) 
    ds = ds.assign_coords(time=conv_obj['time'])
    ds = ds.load()
    print('removing temp files ..')
    for path_year in path_years:                                                            # remove temp_calc files after files have been combined
        os.remove(path_year)
    return ds['rome']



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('rome test starting')
    import os
    import sys
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data             as mD
    import dimensions_data          as dD
    import var_calc.conv_obj_var    as cO

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.show_plots  as sP
    import get_plot.line_plot   as lP
    import get_plot.map_plot    as mP


    switch = {
        'test_sample':  False, 
        'fixed_area':   False,
        'from_scratch_calc':    True,  're_process_calc':  False,  # if both are true the scratch file is replaced with the reprocessed version (only matters for calculated variables / masked variables)
        'from_scratch':         True,  're_process':       False   # same as above, but for base variables
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
    distance_matrix.load()
    # print(distance_matrix)
    # exit()
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
        conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
        obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
        conv_obj.load()
        obj_id.load()
        dim = dD.dims_class(conv_obj)

        # conv_obj = conv_obj.sel(time = slice('1970-01-01', '1971')) # reducing number of workers also reduces chunks
        # obj_id = obj_id.sel(time = slice('1970-01-01', '1971'))

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
        print('ROME results:')
        # [print(f'{dataset}: {ds_rome[dataset].data}') for dataset in ds_rome.data_vars]
        print(ds_rome)
            
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



    # ----------------------------
    #  Call from different script
    # ----------------------------
    # import os
    # import sys
    # sys.path.insert(0, f'{os.getcwd()}/util-calc')
    # import conv_org.rome_calc as rC
    # rome = rC.get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = f'{dataset}_{experiment}') dim: (time)

