'''
# ----------------
#     o_prop
# ----------------
Calculates properties of convective objects, such as; position, area, precipitation, humidity, etc.
Function:
    get_prop_obj_id(prop_name, da_prop, da_area, conv_obj, obj_id)

Input:
    prop            - dict                              {string : boolean}          options given after if __main__
    conv_obj        - integer matrix indifying objects  dim: (time, lat, lon)       get from: util-data/var_calc/conv_obj_var.py
    obj_id          - included objects                  dim: (time, obj)            get from: util-data/var_calc/conv_obj_var.py
    dim             - dimensions (object)                                           get from: util-data/dimensions_data.py
    folder_temp     - folder in scratch to save partial calc

Output:
    o_prop          - matrix                            dim: (time, obj)
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import dask
from distributed import get_client, Client, LocalCluster
import time
import itertools


# --------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #



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
            preferred_workers = 10
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
            # import webbrowser
            # webbrowser.open(f'{client.dashboard_link}') 
            print(f"Dask Dashboard is available at {client.dashboard_link}")
        result = func(*args, **kwargs)
        if client_exists and initial_workers != preferred_workers:  # scale client back to what it was
            print(f'Changing back number of workers to initial state..')
            client.cluster.scale(initial_workers)
            print(client)
        # if not client_exists:
        #     print(f'closing client, as no initial client was given - does not close for the moment')
            # client.close()
        return result
    return wrapper



# ---------------
#  obj property
# ---------------
def get_prop(prop_name, scene_o, da_prop_day, da_area):
    if prop_name == 'area':
        o_prop = (da_prop_day * scene_o).sum()  # don't weight object area by object area    
    else:
        o_prop = (da_prop_day * da_area * scene_o).sum() / (da_area * scene_o).sum()
    return o_prop

def get_o_prop(prop_name, conv_obj_day, labels, da_prop_day, da_area):
    o_prop = []
    for _, label_o in enumerate(labels): 
        scene_o = conv_obj_day.isin(label_o)
        scene_o = xr.where(scene_o > 0, 1, 0)
        # if _ == 0:
        #     print(scene_o)
        #     exit()
        o_prop.append(get_prop(prop_name, scene_o, da_prop_day, da_area)) 
    return o_prop 

def o_prop_day(prop_name, prop_obj_id_year, da_prop_year, da_area, conv_obj_year, obj_id_year, day):
    conv_obj_day = conv_obj_year.isel(time = day)
    labels = obj_id_year.isel(time = day).dropna(dim='obj')
    da_prop_day = da_prop_year.isel(time = day)
    o_prop = get_o_prop(prop_name, conv_obj_day, labels, da_prop_day, da_area)
    prop_obj_id_year[day,:len(labels)] = o_prop

# currently assigning values to daily timesteps of the input matrix
# could also create an xarray like one timestep of prop_obj_id, and then concatenate all the daily timesteps



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
def o_prop_year(data_object_copies, year, indices):
    print(f'processing year: {year} ..')
    prop_obj_id_year, da_prop_year, conv_obj_year, obj_id_year = [da.sel(time = indices) for da in data_object_copies[0:4]]
    da_area = data_object_copies[4]
    prop_name = data_object_copies[5]
    [o_prop_day(prop_name, prop_obj_id_year, da_prop_year, da_area, conv_obj_year, obj_id_year, day) for day in range(len(conv_obj_year.time))]
    return prop_obj_id_year

@ensure_client
@timing_decorator
def get_prop_obj_id(prop_name, da_prop, da_area, conv_obj, obj_id):
    ''' 
    da_prop:       (time, lat, lon) (property values)
    da_area:       (time, lat, lon) (area of gridboxes, for weighting properties)
    conv_obj:      (time, lat, lon) (integer values)
    obj_id:        (time, obj)      (integer values and NaN)
    '''
    print('getting prop_obj_id')
    time_coords = conv_obj['time']
    year_indices_dict = time_index_dict(time_coords, 'year')                                        # get the index for the days in the years 
    conv_obj, obj_id, da_prop = [convert_time_to_index(da) for da in [conv_obj, obj_id, da_prop]]   # convert time to index (cannot be cftime coordinate when scattering)
    prop_obj_id = xr.where(obj_id > 0, 1, np.nan)                                                   # make matrix location with no object into nan
    client = get_client()
    chunk_size = len(client.cluster.workers) # Chunk size (number of years) matching number of workers
    chunks = [dict(itertools.islice(year_indices_dict.items(), i, i + chunk_size)) for i in range(0, len(year_indices_dict), chunk_size)]
    all_results = []
    # print(chunks)
    for _, chunk in enumerate(chunks):
        chunk_indices = list(itertools.chain.from_iterable(chunk.values()))
        data_objects = [prop_obj_id.isel(time=chunk_indices), da_prop.isel(time=chunk_indices), conv_obj.isel(time=chunk_indices), obj_id.isel(time=chunk_indices), da_area, prop_name] 
        data_object_copies = [scatter_data(data_object, client) for data_object in data_objects] # give copy of data to workers to work on independently
        print('scattered data across workers')
        futures = [o_prop_year(data_object_copies, year, indices) for year, indices in chunk.items()]   # task graphs (futures) to be executes in parallel
        print('created task graphs')
        print('started compute')
        results = dask.compute(*futures)                        # ask workers to execute list of futures
        print('finished compute')
        results = list(itertools.chain.from_iterable(results))  # each future in the list becomes a list itself, so flatten the list of lists
        all_results.extend(results)                             # accumulate the results from the chunks
    prop_obj_id = xr.concat(all_results, dim='time')
    prop_obj_id = prop_obj_id.assign_coords(time=time_coords)
    # xr.Dataset({'prop_obj_id ': prop_obj_id}).to_netcdf(path_year, mode="w")
    # print(prop_obj_id )
    return prop_obj_id



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import os
    import sys
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f'running {script_name}')
    import multiprocessing
    ncpus = int(os.getenv('PBS_NCPUS', multiprocessing.cpu_count()))
    print(f'Available resources: {ncpus} CPUs \n (check resource allocation script for available memory)')

    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data             as mD
    import dimensions_data          as dD
    import variable_calc            as vC

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot        as mP
    import get_plot.show_plots      as sP

    switch_prop = {
        'area': True,
        'lat':  False,
        'lon':  False,
        }

    switch = {
        'test_sample':  False, 
        'fixed_area':   False,
        'from_scratch': True,   're_process':   False,  # if taking data from scratch or calculating
        }

    switch_test = {
        'remove_test_plots':            False,          # remove all plots in zome_plots
        'remove_script_plots':          False,          # remove plots generated from this script
        'plot_objects':                 False,          # plot 1
        'plot_da_prop':                 False,          # plot 2
        'plot_object_prop':             False,          # plot 3
        'calc_obj_prop':                True,          # calculate metric
        }
    
    sP.remove_test_plots()                                                      if switch_test['remove_test_plots']     else None
    sP.remove_folder_plots(folder = f'{os.getcwd()}/zome_plots/{script_name}')  if switch_test['remove_script_plots']   else None    
    experiment = cD.experiments[0]
    for prop_name in [k for k, v in switch_prop.items() if v]:
        print(f'object property: {prop_name}')
        ds_obj_snapshot = xr.Dataset()
        ds_prop_snapshot = xr.Dataset()
        ds_prop_obj_snapshot = xr.Dataset()
        ds_prop_obj_id = xr.Dataset()
        for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
            print(f'\t dataset: {dataset}')
            # ----------------------------------------------------------------------------------- Get data -------------------------------------------------------------------------------------------------- #
            conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
            conv_obj.load()
            obj_id.load()
            dim = dD.dims_class(conv_obj)
            da_area = dim.aream

            # conv_obj = conv_obj.sel(time = slice('1970-01-01', '1979')) # When submitting jobs or working in interactive job, the scheduler can handle scattering of larger data arrays
            # obj_id = obj_id.sel(time = slice('1970-01-01', '1979'))     # the size of the scattered data array can also be handled by the chunk_size variable (or equivalently by the number of workers assigned - more workers larger dataset)
            
            if prop_name == 'area':
                da = dim.aream
                ntimes, lat, lon = conv_obj.shape
                da_replicated = np.repeat(da[np.newaxis, :, :], ntimes, axis=0)
                da_prop = xr.DataArray(da_replicated, dims=conv_obj.dims, coords=conv_obj.coords)

            if prop_name == 'lat':
                da = dim.latm
                ntimes, lat, lon = conv_obj.shape
                da_replicated = np.repeat(da[np.newaxis, :, :], ntimes, axis=0)
                da_prop = xr.DataArray(da_replicated, dims=conv_obj.dims, coords=conv_obj.coords)
                # print(da_prop)
                # exit()

            # conv_obj = conv_obj.sel(time = slice('1970-01-01', '1971'))
            # obj_id = obj_id.sel(time = slice('1970-01-01', '1971'))
            # da_prop = da_prop.sel(time = slice('1970-01-01', '1971'))

            # print(f'conv_obj: \n {conv_obj}')
            # print(f'obj_id: \n {obj_id}')
            # print(f'da_prop: \n {da_prop}')
            # print(f'timeaxis: \n {conv_obj.time}')
            # exit()
                
            
            # ------------------------------------------------------------------------------------ Calculate --------------------------------------------------------------------------------------------------- #     
            if switch_test['plot_objects']:
                ds_obj_snapshot[dataset] = xr.where(conv_obj.isel(time = 0) > 0, 1, 0)

            if switch_test['plot_da_prop']:
                ds_prop_snapshot[dataset] = da_prop.isel(time = 0)

            if switch_test['plot_object_prop']:
                conv_obj_day = xr.where(conv_obj.isel(time = 0) > 0, 1, np.nan)
                da_prop_day = da_prop.isel(time = 0)
                ds_prop_obj_snapshot[dataset] = da_prop_day * conv_obj_day
                    
            if switch_test['calc_obj_prop']:
                ds_prop_obj_id[dataset] = get_prop_obj_id(prop_name, da_prop, da_area, conv_obj, obj_id)
        

        # ------------------------------------------------------------------------------------------- Plot -------------------------------------------------------------------------------------------------- #
        if switch_test['plot_objects']:
            ds = ds_obj_snapshot
            label = 'convection [0,1]'
            vmin = 0
            vmax = 1
            cmap = 'Greys'
            title = 'conv_obj'
            filename = f'a_{title}.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = title, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

        if switch_test['plot_da_prop']:
            ds = ds_prop_snapshot
            label = f'{prop_name}'
            vmin = None #0
            vmax = None #20
            cmap = 'Blues'
            filename = f'b_{prop_name}_scene.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

        if switch_test['plot_object_prop']:
            ds = ds_prop_obj_snapshot
            label = f'{prop_name}'
            vmin = None
            vmax = None
            cmap = 'Blues'
            filename = f'c_object_{prop_name}.png'
            fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

        if switch_test['calc_obj_prop']:
            print('obj_prop results:')
            print(ds_prop_obj_id)
            print(ds_prop_obj_id[dataset].isel(time = slice(0,2)))



    # ----------------------------
    #  Call from different script
    # ----------------------------
    # import os
    # import sys
    # sys.path.insert(0, f'{os.getcwd()}/util-calc')
    # import conv_obj.o_prop as oP
    # rome = rC.get_rome(conv_obj, obj_id, distance_matrix, dim, folder_temp = f'{dataset}_{experiment}') dim: (time)
