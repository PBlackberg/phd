''' 
# --------------------------------
#  Conv_org - Convective regions
# --------------------------------
This script generates objects to be considered convective.
The convective regions are identified as precipitation rates exceeding the 95th percentile (or other percentile) precipitation rate
Typically the preciptiation threshold ranges from 17-19 mm / day

to use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  
sys.path.insert(0, f'{os.getcwd()}/util-data')
import var_calc.conv_obj_var    as cOs

switch = {'fixed_area': False}
conv = cO.get_conv_var(switch, dataset, experiment)
conv_obj, obj_id = dO.get_conv_obj(conv)
'''


# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import skimage.measure as skm
import dask
from distributed import get_client, Client, LocalCluster
import itertools


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_base as vB



# ---------------------
#    Dask related
# ---------------------
def ensure_client(func):
    ''' If a function that uses dask is called from a different script, the client can temporarily be scaled 
        and then reverted back to what it was '''
    def wrapper(*args, **kwargs):
        client_exists = True
        try:
            client = get_client()
            initial_workers = len(client.cluster.workers)
        except ValueError:
            print(f'No initial client \n creating client..')
            client_exists = False
            cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='8GB')
            client = Client(cluster)
            print(client)
            print(f"Dask Dashboard is available at {client.dashboard_link}")
        if client_exists and initial_workers != 4:  # scale client to set number of workers
            print(f'Changing number of workers temporarily..')
            client.cluster.scale(4)
            print(client)
        result = func(*args, **kwargs)
        if client_exists:                           # scale client back to what it was
            print(f'Changing back number of workers to initial state..')
            client.cluster.scale(initial_workers)
            print(client)
        if not client_exists:
            print(f'closing client, as no initial client was given')
            client.close()
        return result
    return wrapper



# ---------------------
#  Convective regions
# ---------------------
# ----------------------------------------------------------------------------------- Get convective regions --------------------------------------------------------------------------------------------------- #
def get_precipitation(switch, dataset, experiment, resolution):
    print('executes')
    da = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily') #.isel(time = slice(0, 2))
    da.load()
    return da

def get_conv_threshold(da, percentile):
    conv_threshold = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
    return conv_threshold

def get_fixed_area_conv_threshold(da, percentile):
    return da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)

def get_conv_pr(switch, dataset, experiment, resolution, percentile):
    ''' Binary mask where precipitation exceeds threshold (convective threshold) '''
    da = get_precipitation(switch, dataset, experiment, resolution)
    if switch.get('fixed_area', False): # if 'fixed_area' is an option and is true
        conv_threshold = get_fixed_area_conv_threshold(da, percentile)
    else:
        conv_threshold = get_conv_threshold(da, percentile)
    conv_regions = (da > conv_threshold)*1
    return conv_regions


# -------------------------------------------------------------------------------- Get connected components (objects) --------------------------------------------------------------------------------------------------- #
def connect_boundary(da):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) '''
    s = np.shape(da)
    for row in np.arange(0,s[0]):
        if da[row,0]>0 and da[row,-1]>0:
            da[da==da[row,0]] = min(da[row,0],da[row,-1])
            da[da==da[row,-1]] = min(da[row,0],da[row,-1])
    return da

def get_conv_obj(switch, dataset, experiment, resolution, percentile):
    print('finding convective objects')
    da = get_conv_pr(switch, dataset, experiment, resolution, percentile)
    conv_obj = xr.zeros_like(da)
    obj_dim_size = (len(da.lat) * len(da.lon))/2  # Upper limit of number of objects in a timestep
    dim_list = []
    obj_id = xr.DataArray(np.nan, dims=('time', 'obj'), coords={'time': da.time, 'obj': np.arange(obj_dim_size)})
    for timestep in np.arange(0,len(da.time)):
        if timestep % 365 == 0:
            print(f'\t Processing time: {str(da.time.data[timestep])[:-8]} ...')
        labels_np = skm.label(da.isel(time=timestep).values, background=0, connectivity=2) # returns numpy array
        labels_np = connect_boundary(labels_np)
        conv_obj[timestep,:,:] = labels_np
        labels = np.unique(labels_np)[1:]  # first unique value (zero) is background
        obj_id[timestep, :len(labels)] = labels
        dim_list.append(len(labels))
    obj_id = obj_id.sel(obj = slice(0, max(dim_list)-1)) # remove excess nan (sel is inclusive at end, so need to subtract 1)
    print('finished convective objects')
    return conv_obj, obj_id


# ------------------------------------------------------------------------------------ Pick subset of objects --------------------------------------------------------------------------------------------------- #
def create_mock_metric_mask(obj_id):
    ''' 
    This function picks out a subset of obj_id from a similar (time, obj) matrix 

    Use case:
    let's say the goal is to plot objects exceeding a particular area. 
    If the area of objects is calculated (stack scenes with individual objects multiply with area and sum spatial dimensions)
    Then a matrix (time, obj) describing the area can be created.
    Putting a threshold on this area matrix creates a binary matrix as a mask.
    If the mask is applied to obj_id, the objects satisfying the area condition can be identified.
    '''
    metric_id_mask = xr.zeros_like(obj_id)
    metric_id_mask[0,0:8] = [1]*8
    metric_id_mask[1,0:4] = [1]*4
    metric_id_mask = metric_id_mask.where(metric_id_mask > 0, np.nan)
    return metric_id_mask

def subset_day(conv_obj, obj_id_masked, timestep):
    conv_obj_ts = conv_obj.isel(time = timestep)
    obj_id_masked_ts = obj_id_masked.isel(time = timestep)
    mask = conv_obj_ts.isin(obj_id_masked_ts)
    subset_day = conv_obj_ts.where(mask, other = 0)
    return subset_day

@dask.delayed
def subset_year(conv_obj, obj_id_masked, year):
    conv_obj_year = conv_obj.sel(time = f'{year}')
    obj_id_masked_year = obj_id_masked.sel(time = f'{year}')
    subset_year = [subset_day(conv_obj_year, obj_id_masked_year, day) for day in range(len(conv_obj_year.time))]
    return subset_year

# @ensure_client
def get_obj_subset(conv_obj, obj_id, metric_id_mask):
    ''' 
    conv_obj:      (time, lat, lon) (integer values)
    obj_id:        (time, obj)      (integer values and NaN)
    metric_mask:   (time, obj)      ([NaN, 1])
    '''
    print('getting subset matrix')
    client = get_client()
    obj_id_masked = obj_id * metric_id_mask
    # conv_obj = conv_obj.chunk({'time':'auto'})
    # print(conv_obj)
    conv_obj_copies = client.scatter(conv_obj, direct=True, broadcast=True)             #.chunk({'time': 'auto'})
    obj_id_masked_copies = client.scatter(obj_id_masked, direct=True, broadcast=True)
    futures = [subset_year(conv_obj_copies, obj_id_masked_copies, year) for year in np.unique(conv_obj['time'].dt.year)] # list of futures
    results = dask.compute(*futures)                        # each future becomes a list
    results = list(itertools.chain.from_iterable(results))  # flatten the list of lists into one list
    conv_obj_subset = xr.concat(results, dim='time')
    print(conv_obj_subset)
    return conv_obj_subset

# @dask.delayed
# def subset_year(conv_obj_year, obj_id_masked_year):
#     subset_year = [subset_day(conv_obj_year, obj_id_masked_year, day) for day in range(len(conv_obj_year.time))]
#     return subset_year

# def get_obj_subset(conv_obj, obj_id, metric_id_mask):
#     ''' 
#     conv_obj:      (time, lat, lon) (integer values)
#     obj_id:        (time, obj)      (integer values and NaN)
#     metric_mask:   (time, obj)      ([NaN, 1])
#     '''
#     print('getting subset matrix')
#     obj_id_masked = obj_id * metric_id_mask
#     futures = []
#     for year in np.unique(conv_obj['time'].dt.year):
#         conv_obj_year = conv_obj.sel(time = f'{year}')
#         obj_id_masked_year = obj_id_masked.sel(time = f'{year}')
#         futures.append(subset_year(conv_obj_year, obj_id_masked_year))
#     print(np.shape(futures))
#     result1 = dask.compute(*futures[0:3])
#     result2 = dask.compute(*futures[3:])
#     print(np.shape(result1))
#     print(np.shape(result2))
#     result = list(itertools.chain.from_iterable(result1)) + list(itertools.chain.from_iterable(result2))
#     conv_obj_subset = xr.concat(result, dim='time')
#     print(conv_obj_subset)
#     return conv_obj_subset



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('Convective region test starting')

    def create_client():
        cluster = LocalCluster(n_workers=4, threads_per_worker=10, memory_limit='8GB')
        client = Client(cluster)
        print(client)
        print(f"Dask Dashboard is available at {client.dashboard_link}")
        import webbrowser
        webbrowser.open(f'{client.dashboard_link}') 
        # print('for dashboard, open: http://127.0.0.1:8787/status') # this is the link that is given when opening the browser from the login node
        return client
    client = create_client()

    import missing_data as mD
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  
    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot         as mP


    switch = {
        'test_sample':    False,
        'fixed_area':     False
        }

    switch_test = {
        'delete_previous_plots':        False,
        'get_conv_obj_from_file':       True,
        'plot_conv':                    True,
        'plot_obj':                     True,
        'plot_obj_subset':              True,
        'plot_obj_subset_metric':       True, # haven't done yet (need to calculate area_obj_id)
        }
    
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
    percentile = int(cD.conv_percentiles[0])*0.01
    resolution = cD.resolutions[0]
    experiment = cD.experiments[0]
    timestep = 1
    
    ds_conv = xr.Dataset()
    ds_conv_obj = xr.Dataset()
    ds_conv_obj_subset = xr.Dataset()
    for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
        print(f'dataset: {dataset}')
        # -------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        if switch_test['get_conv_obj_from_file']:
            import variable_calc as vC
            conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = resolution, timescale = 'daily', from_folder = True, re_process = False)            
            obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = resolution, timescale = 'daily', from_folder = True, re_process = False)
            conv_obj = conv_obj.sel(time = slice('1970', '1975'))
            obj_id = obj_id.sel(time = slice('1970', '1975'))
            conv_obj.load()
            obj_id.load()
            conv = (conv_obj > 0)*1
            # print(conv_obj)
            # print(obj_id)
            # print(conv_obj)
            metric_id_mask = create_mock_metric_mask(obj_id)
            conv_obj_subset = get_obj_subset(conv_obj, obj_id, metric_id_mask)
            # print(conv)
            # print(type(conv))
            # exit()
        else:
            conv = get_conv_pr(switch, dataset, experiment, resolution, percentile)
            conv_obj, obj_id = get_conv_obj(switch, dataset, experiment, resolution, percentile)
            metric_id_mask = create_mock_metric_mask(obj_id)
            conv_obj_subset = get_obj_subset(conv_obj, obj_id, metric_id_mask)

        # print(f'conv: \n {conv}')
        # print(f'conv_obj: \n {conv_obj}')
        # print(f'obj_id: \n {obj_id}')
        # print(f'metric id mask: \n {metric_id_mask}')
        # print(f'metric id mask: \n {conv_obj_subset}')
        # print(f'unique subset: \n {np.unique(conv_obj_subset)}')
        # exit()


        # ------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
        if switch_test['plot_conv']:
            ds_conv[dataset] = conv.isel(time = timestep)

        if switch_test['plot_obj']:
            ds_conv_obj[dataset] = conv_obj.isel(time = timestep)

        if switch_test['plot_obj_subset']:
            ds_conv_obj_subset[dataset] = conv_obj_subset.isel(time = timestep)
            

    # -------------------------------------------------------------------------------------------- plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_conv']:
        ds = ds_conv
        label = 'conv [0,1]'
        vmin = 0
        vmax = 1
        cmap = 'Greys'
        title = 'convective_regions'
        filename = f'{title}.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_obj']:
        ds = ds_conv_obj
        label = 'conv [integer]'
        vmin = 0
        vmax = len(obj_id.obj)
        cmap = 'Greys'
        title = 'convective_objects'
        filename = f'{title}.png'
        cmap = 'Greys'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()), cat_cmap = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_obj_subset']:
        ds = ds_conv_obj_subset
        label = 'conv [integer]'
        vmin = 0
        vmax = len(obj_id.obj)
        cmap = 'Greys'
        title = 'convective_objects_subset'
        filename = f'{title}.png'
        cmap = 'Greys'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()), cat_cmap = True)
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
    client.close()





