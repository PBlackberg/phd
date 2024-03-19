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
import var_calc.conv_obj_var    as cO

switch = {'fixed_area': False}
conv = cO.get_conv_var(switch, dataset, experiment)
conv_obj, obj_id = dO.get_conv_obj(conv)
'''


# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import skimage.measure as skm
from distributed import Client


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_base as vB



# ---------------------
#  Convective regions
# ---------------------
# ----------------------------------------------------------------------------------- Get convective regions --------------------------------------------------------------------------------------------------- #
def get_precipitation(switch, dataset, experiment, resolution):
    da = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily') #.isel(time = slice(0, 2))
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
    client = Client()
    print(client)
    da = get_conv_pr(switch, dataset, experiment, resolution, percentile)
    conv_obj = xr.zeros_like(da)
    obj_dim_size = (len(da.lat) * len(da.lon))/2  # Upper limit of number of objects in a timestep
    dim_list = []
    obj_id = xr.DataArray(np.nan, dims=('time', 'obj'), coords={'time': da.time, 'obj': np.arange(obj_dim_size)})
    print(da)
    for timestep in np.arange(0,len(da.time)):
        print(f'\t Processing time: {str(da.time.data[timestep])[:-8]} ...')
        labels_np = skm.label(da.isel(time=timestep).values, background=0, connectivity=2) # returns numpy array
        labels_np = connect_boundary(labels_np)
        conv_obj[timestep,:,:] = labels_np
        labels = np.unique(labels_np)[1:]  # first unique value (zero) is background
        obj_id[timestep, :len(labels)] = labels
        dim_list.append(len(labels))
    obj_id = obj_id.sel(obj = slice(0, max(dim_list)-1)) # remove excess nan (sel is inclusive at end, so need to subtract 1)
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

def get_obj_subset(conv_obj, obj_id, metric_id_mask):
    ''' 
    conv_obj:      (time, lat, lon) (integer values)
    obj_id:        (time, obj)      (integer values and NaN)
    metric_mask:   (time, obj)      ([NaN, 1])
    '''
    obj_id_masked = obj_id * metric_id_mask
    conv_obj_subset = xr.zeros_like(conv_obj)
    for timestep in range(len(conv_obj.time)):
        conv_obj_ts = conv_obj.isel(time = timestep)
        obj_id_masked_ts = obj_id_masked.isel(time = timestep)
        mask = conv_obj_ts.isin(obj_id_masked_ts)
        conv_obj_subset[timestep, :, :] = conv_obj_ts.where(mask, other = 0)
    return conv_obj_subset



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('Convective region test starting')
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  
    import missing_data as mD
    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot         as mP

    switch = {
        'test_sample':    False,
        'fixed_area':     False
        }

    switch_test = {
        'delete_previous_plots':        False,
        'plot_conv':                    False,
        'plot_obj':                     False,
        'plot_obj_subset':              False,
        'plot_obj_subset_metric':       False, # haven't done yet
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
        conv_obj, obj_id = get_conv_obj(switch, dataset, experiment, resolution, percentile)
        conv = get_conv_pr(switch, dataset, experiment, resolution, percentile)

        conv_obj.load()
        obj_id.load()
        conv.load()

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
            print('calculating convective regions')
            ds_conv[dataset] = conv.isel(time = timestep)

        if switch_test['plot_obj']:
            print(f'calculating convective objects')
            # print(conv_obj.isel(time = timestep))
            # print(np.unique(conv_obj.isel(time = timestep)))
            # exit()
            ds_conv_obj[dataset] = conv_obj.isel(time = timestep)

        if switch_test['plot_obj_subset']:
            print(f'calculating subset of convective objects')
            ds_conv_obj_subset[dataset] = conv_obj_subset.isel(time = timestep)
            

    # -------------------------------------------------------------------------------------------- plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_conv']:
        print('plotting convective regions')
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
        print('plotting convective objects')
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
        print('plotting subset of convective objects')
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

