''' 
# ------------------------
#  Precipitation metrics
# ------------------------
In this script organization metrics are calculated from convective regions.
The convective regions are binary 2D fields.
The convective regions are determined by precipitation rates exceeding a percentile threshold (on average 5% of the domain covered is the default)
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import skimage.measure as skm


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV
import myFuncs as mF



# ------------------------
#  Calculation: Metrics
# ------------------------
@mF.timing_decorator()
def get_snapshot(da):
    plot = False
    if plot:
        import myFuncs_plots as mFd     
        for timestep in np.arange(0, len(da.time.data)):
            fig = mFd.plot_scene(da.isel(time=timestep), ax_title = timestep, vmin = 0, vmax = 20)    #, vmin = 0, vmax = 60) #, cmap = 'RdBu')
            if mFd.show_plot(fig, show_type = 'cycle', cycle_time = 0.5):        # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)
                break
    return da.isel(time=0) 

@mF.timing_decorator()
def get_tMean(da):
    return da.mean(dim='time', keep_attrs=True)

@mF.timing_decorator()
def get_sMean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(dim=('lat','lon'), keep_attrs=True).compute() # dask objects require the compute part



# --------------------------
# Calculation: Metric types
# --------------------------
# -------------------------------------------------------------------------------------------- Percentile based ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def calc_percentile(da, percentile):
    return da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)

@mF.timing_decorator()
def get_da_percentile(da, percentile):
    percentile_value = calc_percentile(da, percentile)
    return da.where(da >= percentile_value)


# ------------------------------------------------------------------------------------------------ Rxday ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def get_rxday(da, nb_days = '5'):
    ''' Most extreme daily gridpoint value locally over set time period (1-year here) '''
    if int(nb_days) > 1:
        da = da.resample(time=f'{nb_days}D').mean(dim='time')
    da = da.resample(time='Y').max(dim='time')
    return da


# -------------------------------------------------------------------------------------------- Object oriented ----------------------------------------------------------------------------------------------------- #
def calc_conv_threshold(switch, da):
    ''' Convection can be based on fixed precipitation rate threshold (variable area) or fixed areafraction (fixed area). Both applications have the same mean area for the complete time period'''
    if switch['fixed_area']:
        conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
    else:
        conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
        conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
    return conv_threshold

@mF.timing_decorator()
def get_pr_o_day(switch, da, dim):
    ''' Precipitation rate in each contigous convective region (object) '''
    conv_threshold = calc_conv_threshold(switch, da)
    pr_o, labels_o = [], []
    for day in np.arange(0,len(da.time.data)):
        print(f'\t Processing ... currently on {da.time.data[day].year}') if day % 365 == 0 else None
        pr_day = da.isel(time=day)
        pr_day3d = np.expand_dims(pr_day,axis=2)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        L = mF.connect_boundary(L)
        labels_oScene = np.unique(L)[1:] # first number is background (0)
        obj3d = np.stack([(L==label) for label in labels_oScene],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        pr_oScene = np.sum(obj3d * pr_day3d * dim.aream3d, axis=(0,1)) / np.sum(obj3d * dim.aream3d, axis=(0,1))
        pr_o.append(pr_oScene)          # list of lists
        # labels_o.append(labels_oScene)  # list of lists # can make this a separate matrix in org_calc
    return pr_o

def create_o_list(list_of_lists):
    list_o = [item for sublist in list_of_lists for item in sublist]
    return xr.DataArray(data = list_o, dims = ("obj"), coords = {"obj": np.arange(0, len(list_o))})

@mF.timing_decorator()
def create_o_array(da, list_of_lists):
    max_length = max(len(sublist) for sublist in list_of_lists)
    array_filled = np.full((len(list_of_lists), max_length), np.nan)
    for i, sublist in enumerate(list_of_lists):
        array_filled[i, :len(sublist)] = sublist
        array = xr.DataArray(data = array_filled, dims = ("time", "obj"), coords = {"time": da.time.data})
    return array


# -------------------------------------------------------------------------------------------- Organization proxy ----------------------------------------------------------------------------------------------------- #
# @mF.timing_decorator
# def calc_F_pr10(da):
#     ''' Frequency of gridboxes exceeding 10 mm/day on monthly [% of domain]'''
#     da = da.resample(time='1MS').mean(dim='time')
#     mask = xr.where(da>10,1,0)
#     F_pr10 = (mask.sum(dim=('lat','lon')) / (len(da['lat']) * len(da['lon']))) * 100
#     return F_pr10



# ------------------------
#   Run / save metrics
# ------------------------
# --------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def fix_metric_type_name(switch, metric_type_name):
    if metric_type_name in ['pr_o', 'pr_o_array']:
        metric_type_name = f'{metric_type_name}_{mV.conv_percentiles[0]}thprctile'
        metric_type_name = f'{metric_type_name}_fixed_area' if switch['fixed_area'] else metric_type_name
    return metric_type_name

def calc_metric_type(switch_metric, switchM, switch, da):
    dim = mF.dims_class(da)
    metric_type, metric_type_name = None, None
    for metric_type_name in [k for k, v in switch_metric.items() if v]:
        print(f'Running {metric_type_name}')
        metric_type = None
        metric_type = da                                                                                if metric_type_name in ['pr']                                                       else metric_type
        metric_type = calc_percentile(da, percentile = int(metric_type_name.split('_')[1]) * 0.01)      if metric_type_name in ['pr_90', 'pr_95', 'pr_97', 'pr_99'] and switchM['other']    else metric_type
        metric_type = get_da_percentile(da, percentile = int(metric_type_name.split('_')[1]) * 0.01)    if metric_type_name in ['pr_90', 'pr_95', 'pr_97', 'pr_99']                         else metric_type
        metric_type = get_rxday(da, nb_days = metric_type_name[5])                                      if metric_type_name in ['pr_rx1day', 'pr_rx5day']                                   else metric_type                                      
        metric_type = create_o_list(get_pr_o_day(switch, da, dim))                                      if metric_type_name in ['pr_o']                                                     else metric_type
        metric_type = create_o_array(da, get_pr_o_day(switch, da, dim))                                 if metric_type_name in ['pr_o_array']                                               else metric_type
        metric_type_name = fix_metric_type_name(switch, metric_type_name)
        yield metric_type, metric_type_name

def calc_metric(switch_metric, switchM, switch, metric_type, metric_type_name):
    metric, metric_name = None, None
    for metric_name in [k for k, v in switchM.items() if v]:
        metric = None
        metric, metric_name = [metric_type, metric_type_name]                               if metric_name == 'other'       else [metric, metric_name]
        metric, metric_name = [get_snapshot(metric_type), f'{metric_type_name}_snapshot']   if metric_name == 'snapshot'    else [metric, metric_name]
        metric, metric_name = [get_sMean(metric_type), f'{metric_type_name}_sMean']         if metric_name == 'sMean'       else [metric, metric_name]
        metric, metric_name = [get_tMean(metric_type), f'{metric_type_name}_tMean']         if metric_name == 'tMean'       else [metric, metric_name]
        yield metric, metric_name


# --------------------------------------------------------------------------------------------------- Run metric ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def run_pr_metrics(switch_metric, switchM, switch):
    print(f'variable: daily precipitation field')
    print(f'metric_type: {[key for key, value in switch_metric.items() if value]}')
    print(f'metric: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for dataset, experiment in mF.run_dataset():
        da = mF.load_variable({'pr': True}, switch, dataset, experiment)
        for metric_type, metric_type_name in calc_metric_type(switch_metric, switchM, switch, da):
            for metric, metric_name in calc_metric(switch_metric, switchM, switch, metric_type, metric_type_name):
                # print(metric_name)
                # print(metric)
                path = mF.save_metric(switch, 'pr', dataset, experiment, metric, metric_name) if metric is not None else None
                # print(path)
                # ds = xr.open_dataset(path)
                # print(ds)



# ------------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_metric = {                                                                                               # metrics (can choose multiple)
        'pr':           True,  'pr_90':         False,  'pr_95':    False,  'pr_97':    False,  'pr_99':    False,  # percentiles
        'pr_rx1day':    False, 'pr_rx5day':     False,                                                              # local extremes
        'pr_o':         False, 'pr_o_array':    True,                                                              # object precipitation
        'F_pr10':       False,                                                                                      # organization proxy
        }
    
    switchM = {                                                                                     # metric tyoe (has to work for all metrics chosen in switch_metric)
        'snapshot': True,   'sMean':    False,  'tMean':    False,  'other':    False
        }
    
    switch = {                                                                              # settings
        'constructed_fields':   False,  'test_sample':      False,                          # dataset type
        'fixed_area':           False,                                                      # conv_threshold
        'save_folder_desktop':  False,  'save':             False,   'save_scratch': False,  # save (can only do one at a time)
        }
    
    run_pr_metrics(switch_metric, switchM, switch)



