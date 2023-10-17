import numpy as np
import xarray as xr
import skimage.measure as skm
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import constructed_fields as cF
import get_data as gD
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myClasses as mC
import myFuncs as mF



# ------------------------
#       Get data
# ------------------------
# --------------------------------------------------------------------------------------------- Load data ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def load_data(switch, source, dataset, experiment):
    if switch['constructed_fields']:
        da = cF.var3D
    if switch['sample_data']:
        dataset_alt = 'GPCP' if dataset in ['GPCP_1998-2009', 'GPCP_2010-2022'] else dataset
        da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/pr/{source}/{dataset_alt}_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['pr']
    if switch['gadi_data']:
        da = gD.get_var_data(source, dataset, experiment, 'pr')

    # There is a double trend in high percentile precipitation rate for the first 12 years of the data, so
    if dataset == 'GPCP_1998-2009':
        da = da.sel(time= slice('1998', '2009'))
    if dataset == 'GPCP_2010-2022':
        da = da.sel(time= slice('2010', '2022'))
    return da


# ----------------------------------------------------------------------------------------- find conv threshold ----------------------------------------------------------------------------------------------------- #
def calc_conv_threshold(switch, da):
    ''' Convection can be based on fixed precipitation rate threshold (variable area) or fixed areafraction (fixed area). Both applications have the same mean area for the complete time period'''
    if switch['fixed_area']:
        conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
    else:
        conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
        conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
    return conv_threshold



# ------------------------
#    Calculate metrics
# ------------------------
# ------------------------------------------------------------------------------------------ Mean precipitation ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def get_pr_snapshot(da):
    pr_snapshot = da.isel(time=0) 
    return pr_snapshot

@mF.timing_decorator
def calc_pr_sMean(da):
    aWeights = np.cos(np.deg2rad(da.lat))
    pr_sMean = da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True) 
    return pr_sMean

@mF.timing_decorator
def get_pr_tMean(da):
    pr_tMean = da.mean(dim='time')
    return pr_tMean

@mF.timing_decorator
def calc_pr_area(da):
    ''' Area covered in domain [% of domain]'''
    dim = mC.dims_class(da)
    da = xr.where(da>0, 1, 0)
    area = np.sum(da * np.transpose(dim.aream3d, (2, 0, 1)), axis=(1,2)) / np.sum(dim.aream)
    return area


# -------------------------------------------------------------------------------------------- Global extremes ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def get_percentile_snapshot(da, percentile):
    da_snapshot = da.isel(time=0)
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    percentile_snapshot = da_snapshot.where(da_snapshot>=percentile_value)
    return percentile_snapshot

@mF.timing_decorator
def calc_percentile(da, percentile):
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return percentile_value

@mF.timing_decorator
def calc_percentile_sMean(da, percentile):
    aWeights = np.cos(np.deg2rad(da.lat))
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    meanInPercentile = da.where(da >= percentile_value).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)
    return meanInPercentile


# --------------------------------------------------------------------------------------------- Local extremes ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def get_rx1day_snapshot(da):
    ''' Most extreme daily gridpoint value locally over set time period (1-year here) '''
    rx1day_snapshot = da.resample(time='Y').max(dim='time').isel(time=0)
    return rx1day_snapshot

@mF.timing_decorator
def get_rx1day_tMean(da):
    ''' Most extreme daily gridpoint value locally over set time period (1-year here) '''
    rx1day = da.resample(time='Y').max(dim='time')
    rx1day_tMean = rx1day.mean(dim='time')
    return rx1day_tMean

@mF.timing_decorator
def calc_rx1day_sMean(da):
    ''' Most extreme daily gridpoint value locally over set time period (1-year here), spatially averaged '''
    aWeights = np.cos(np.deg2rad(da.lat))
    rx1day = da.resample(time='Y').max(dim='time')
    rx1day_sMean = rx1day.weighted(aWeights).mean(dim=('lat', 'lon')) 
    return rx1day_sMean

@mF.timing_decorator
def get_rx5day_snapshot(da):
    ''' Most extreme 5-day accumulation of daily gridpoint value locally over set time period (1-year here) '''
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day_snapshot = da5day.resample(time='Y').max(dim='time').isel(time=0)
    return rx5day_snapshot

@mF.timing_decorator
def get_rx5day_tMean(da):
    ''' Most extreme 5-day accumulation of daily gridpoint value locally over set time period (1-year here) '''
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    rx5day_tMean = rx5day.mean(dim='time')
    return rx5day_tMean

@mF.timing_decorator
def calc_rx5day_sMean(da):
    ''' Most extreme 5-day accumulation of daily gridpoint value locally over set time period (1-year here), spatially averaged '''
    aWeights = np.cos(np.deg2rad(da.lat))
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    rx5day_sMean = rx5day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx5day_sMean


# --------------------------------------------------------------------------------- convective region based ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def get_pr_o_snapshot(switch, da):
    pr_day = da.isel(time=0)
    conv_threshold = calc_conv_threshold(switch, da)
    L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=0),0)>0, background=0,connectivity=2)
    mF.connect_boundary(L)
    pr_o_snapshot = xr.DataArray(data = L.data, dims = ['lat', 'lon'], coords = {'lat': da.lat.data, 'lon': da.lon.data}, 
                                 attrs = {'units':r'mm day$^-1$', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})
    return pr_o_snapshot

@mF.timing_decorator
def calc_pr_o_sMean(switch, da):
    ''' Precipitation rate in each contigous convective region (object) '''
    conv_threshold = calc_conv_threshold(switch, da)
    dim = mC.dims_class(da)
    o_pr = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        pr_day3d = np.expand_dims(pr_day,axis=2)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_prScene = np.sum(obj3d * pr_day3d * dim.aream3d, axis=(0,1)) / np.sum(obj3d * dim.aream3d, axis=(0,1))
        o_pr = np.append(o_pr, o_prScene)
    o_pr = xr.DataArray(data = o_pr, dims = 'obj', 
                        attrs = {'units':r'mm day$^-1$', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})
    return o_pr



# ------------------------
#   Run / save metrics
# ------------------------
# --------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, source, dataset, experiment, da, metric, metric_type):
    da_calc, metric_name = None, None
    if metric_type == 'snapshot':
        metric_name =f'{metric}_snapshot' 
        da_calc = get_pr_snapshot(da)               if metric == 'pr'        else da_calc
        da_calc = get_percentile_snapshot(da, 0.90) if metric == 'pr_90'     else da_calc
        da_calc = get_percentile_snapshot(da, 0.95) if metric == 'pr_95'     else da_calc
        da_calc = get_percentile_snapshot(da, 0.97) if metric == 'pr_97'     else da_calc
        da_calc = get_percentile_snapshot(da, 0.99) if metric == 'pr_99'     else da_calc
        da_calc = get_rx1day_snapshot(da)           if metric == 'pr_rx1day' else da_calc
        da_calc = get_rx5day_snapshot(da)           if metric == 'pr_rx5day' else da_calc
        da_calc = get_pr_o_snapshot(switch, da)     if metric == 'pr_o'      else da_calc

    if metric_type == 'sMean':
        metric_name =f'{metric}_sMean' 
        da_calc = calc_pr_sMean(da)                 if metric == 'pr'        else da_calc
        da_calc = calc_percentile_sMean(da, 0.90)   if metric == 'pr_90'     else da_calc
        da_calc = calc_percentile_sMean(da, 0.95)   if metric == 'pr_95'     else da_calc
        da_calc = calc_percentile_sMean(da, 0.97)   if metric == 'pr_97'     else da_calc
        da_calc = calc_percentile_sMean(da, 0.99)   if metric == 'pr_99'     else da_calc
        da_calc = calc_rx1day_sMean(da)             if metric == 'pr_rx1day' else da_calc
        da_calc = calc_rx5day_sMean(da)             if metric == 'pr_rx5day' else da_calc
        da_calc = calc_pr_o_sMean(switch, da)       if metric == 'pr_o'      else da_calc

    if metric_type == 'tMean':
        metric_name =f'{metric}_tMean' 
        da_calc = get_pr_tMean(da)                  if metric == 'pr'        else da_calc
        da_calc = get_rx1day_tMean(da)              if metric == 'pr_rx1day' else da_calc
        da_calc = get_rx5day_tMean(da)              if metric == 'pr_rx5day' else da_calc

    if metric_type == 'area':
        metric_name =f'{metric}_area' 
        da_calc = calc_pr_area(da)                  if metric == 'pr'        else da_calc

    if metric_type == '':
        metric_name =f'{metric}'
        da_calc = calc_percentile(da, 0.90)         if metric == 'pr_90'     else da_calc
        da_calc = calc_percentile(da, 0.95)         if metric == 'pr_95'     else da_calc
        da_calc = calc_percentile(da, 0.97)         if metric == 'pr_97'     else da_calc
        da_calc = calc_percentile(da, 0.99)         if metric == 'pr_99'     else da_calc

    metric_name = f'{metric_name}_{mV.conv_percentiles[0]}thprctile' if (metric_name and metric == 'pr_o')                          else metric_name
    metric_name = f'{metric_name}_fixed_area'                        if (metric_name and metric == 'pr_o' and switch['fixed_area']) else metric_name
    mF.save_in_structured_folders(da_calc, f'{mV.folder_save[0]}/metrics', 'pr', metric_name, source, dataset, mV.timescales[0], experiment, mV.resolutions[0])                      if (switch['save'] and da_calc is not None)            else None
    mF.save_file(xr.Dataset(data_vars = {metric_name: da_calc}), f'{home}/Desktop/{metric_name}', f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc') if (switch['save_to_desktop'] and da_calc is not None) else None


# --------------------------------------------------------------------------------------------------- Run metric ----------------------------------------------------------------------------------------------------- #
def run_metric_type(switchM, switch, source, dataset, experiment, da, metric):
    for metric_type in [k for k, v in switchM.items() if v]:
        get_metric(switch, source, dataset, experiment, da, metric, metric_type)

def run_metric(switch_metric, switchM, switch, source, dataset, experiment):
    da = load_data(switch, source, dataset, experiment)
    for metric in [k for k, v in switch_metric.items() if v]:
        run_metric_type(switchM, switch, source, dataset, experiment, da, metric)

def run_experiment(switch_metric, switchM, switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment):
            continue
        print(f'\t\t {experiment}') if experiment else print(f'\t observational dataset')
        run_metric(switch_metric, switchM, switch, source, dataset, experiment)

def run_dataset(switch_metric, switchM, switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'\t{dataset} ({source})')
        run_experiment(switch_metric, switchM, switch, source, dataset)

@mF.timing_decorator
def run_pr_metrics(switch_metric, switchM, switch):
    print(f'Running {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'metric: {[key for key, value in switch_metric.items() if value]}')
    print(f'metric type: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    run_dataset(switch_metric, switchM, switch)


# ------------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_metric = {
        # choose metric
        'pr':                             True,
        'pr_90':                          True,
        'pr_95':                          True,
        'pr_97':                          True,
        'pr_99':                          True,
        'pr_rx1day':                      True,
        'pr_rx5day':                      True,
        'pr_o':                           True,
        }
    
    switchM = {
        # choose type of metric
        'snapshot':                       True,
        'sMean':                          True,
        'tMean':                          True,
        'area':                           True,
        '':                               True,
        }
    
    switch = {
        # Type of dataset
        'constructed_fields':             False, 
        'sample_data':                    True,
        'gadi_data':                      False,

        # conv_threshold
        'fixed_area':                     False,

        # save
        'save':                           True,
        'save_to_desktop':                False
        }
    
    run_pr_metrics(switch_metric, switchM, switch)
    







