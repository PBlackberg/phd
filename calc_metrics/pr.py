import numpy as np
import xarray as xr
import skimage.measure as skm
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF                            # imports common operators
import constructed_fields as cF                 # imports fields for testing
import get_data as gD                           # imports functions to get data from gadi
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                             # imports common variables

# ----------------------------------------------------------------------------------------------- Calculattion ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def calc_rxday_sMean(da):
    ''' Most extreme daily gridpoint value locally over set time period '''
    aWeights = np.cos(np.deg2rad(da.lat))
    rx1day = da.resample(time='Y').max(dim='time')

    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    return rx1day.weighted(aWeights).mean(dim=('lat', 'lon')), rx5day.weighted(aWeights).mean(dim=('lat', 'lon'))

@mF.timing_decorator
def calc_rxday_tMean(da):
    ''' Time-mean of local extremes over set time period '''
    rx1day = da.resample(time='Y').max(dim='time')

    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    return rx1day.mean(dim='time'), rx5day.mean(dim='time')



@mF.timing_decorator
def calc_percentile_snapshot(da, percentile):
    da_snapshot = da.isel(time=0)
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return da_snapshot.where(da_snapshot>=percentile_value).isel(time=0)

@mF.timing_decorator
def calc_meanInPercentile(da, percentile):
    aWeights = np.cos(np.deg2rad(da.lat))
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return da.where(da >= percentile_value).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)



@mF.timing_decorator
def calc_F_pr10_snapshot(da):
    ''' Snapshot of gridboxes exceeding 10 mm/day '''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    return da.isel(time=0).where(mask.isel(time=0) > 0)

@mF.timing_decorator
def calc_F_pr10(da):
    ''' Frequency of gridboxes exceeding 10 mm/day on monthly'''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    return mask.sum(dim=('lat','lon')) * 100 / (len(da['lat']) * len(da['lon']))



@mF.timing_decorator
def calc_o_pr(da, conv_threshold):
    ''' Precipitation rate in each contigous convective region (object) '''
    lat, lon = da['lat'].data, da['lon'].data
    lonm, latm = np.meshgrid(lon,lat)
    dlat, dlon = (lat[1]-lat[0]), (lon[1]-lon[0])
    R = 6371 # km
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    aream3d = np.expand_dims(aream,axis=2) # used for broadcasting
    o_pr = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        pr_day3d = np.expand_dims(pr_day,axis=2)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_prScene = np.sum(obj3d * pr_day3d * aream3d, axis=(0,1)) / np.sum(obj3d * aream3d, axis=(0,1))
        o_pr = np.append(o_pr, o_prScene)
    return o_pr


# ---------------------------------------------------------------------------------------------- Load data ----------------------------------------------------------------------------------------------------- #
def load_data(switch, source, dataset, experiment):
    da = cF.var3D if switch['constructed_fields'] else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/pr/sample_data/{source}/{dataset}_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['pr'] if switch['sample_data'] else da
    da = gD.get_var_data(source, dataset, experiment, 'pr') if switch['gadi_data'] else da
    return da


# ----------------------------------------------------------------------------------------- Put metric in dataset ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, da, metric):
    ds_calc, metric_name = None, None
    if metric == 'rxday_sMean':
        metric_name = 'rxday_pr_sMean'
        rx1day_sMean, rx5day_sMean = calc_rxday_sMean(da)
        ds_calc = xr.Dataset({'rx1day_pr_sMean': rx1day_sMean , 'rx5day_pr_sMean': rx5day_sMean})         
        del rx1day_sMean, rx5day_sMean

    if metric == 'rxday_tMean':
        metric_name = 'rxday_pr_tMean'
        rx1day_tMean, rx5day_tMean = calc_rxday_tMean(da)
        ds_calc = xr.Dataset({'rx1day_pr_tMean': rx1day_tMean , 'rx5day_pr_tMean': rx5day_tMean})
        del rx1day_tMean, rx5day_tMean

    if metric == 'pr_snapshot':
        metric_name = 'pr_snapshot'
        ds_calc = xr.Dataset({metric_name : da.isel(time=0)})

    if metric == 'percentiles_snapshot':
        metric_name = 'percentiles_pr_snapshot'
        ds_calc = xr.Dataset()
        for percentile in [0.95, 0.97, 0.99]:
            ds_calc[f'pr{int(percentile*100)}_snapshot'] = calc_percentile_snapshot(da, percentile)

    if metric == 'percentiles':
        metric_name = 'percentiles_pr'
        ds_calc = xr.Dataset()
        for percentile in [0.95, 0.97, 0.99]:
            ds_calc[f'pr{int(percentile*100)}'] = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
        
    if metric == 'percentiles_sMean':
        metric_name = 'percentiles_pr_sMean'
        ds_calc = xr.Dataset()
        for percentile in [0.95, 0.97, 0.99]:
            ds_calc[f'pr{int(percentile*100)}_sMean'] = calc_meanInPercentile(da, percentile)

    if metric == 'F_pr10_snapshot':
        metric_name = 'F_pr10_snapshot'
        ds_calc = xr.Dataset({metric_name: calc_F_pr10_snapshot(da)})

    if metric == 'F_pr10':
        metric_name = 'F_pr10'
        ds_calc = xr.Dataset({metric_name: calc_F_pr10(da)})

    if metric == 'o_pr':
        metric_name = 'o_pr'
        conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
        conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) if not switch['fixed_area'] else conv_threshold
        o_pr = xr.DataArray(data = calc_o_pr(da, conv_threshold), dims = 'object',
                                attrs = {'units':'mm day' + mF.get_super('-1'),
                                         'descrption': 'area weighted mean precipitation in contiguous convective region (object)'})
        ds_calc = xr.Dataset({metric_name: o_pr})
        del conv_threshold
    return ds_calc, metric_name


# -------------------------------------------------------------------------------- run metric and save ----------------------------------------------------------------------------------------------------- #
def save_metric(source, dataset, experiment, ds, metric_name):
    folder = f'{mV.folder_save[0]}/pr/metrics/{metric_name}/{source}'
    filename = f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc'
    mF.save_file(ds, folder, filename)
    
def run_metric(switch, source, dataset, experiment):
    da = load_data(switch, source, dataset, experiment)
    for metric in [k for k, v in switch.items() if v] : # loop over true keys
        ds, metric_name = get_metric(switch, da, metric)
        save_metric(source, dataset, experiment, ds, metric_name) if switch['save'] and metric_name else None

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if not mF.data_available(source, dataset, experiment):
            continue
        print(f'\t {experiment}') if experiment else print(f'\t observational dataset')
        run_metric(switch, source, dataset, experiment)

def run_dataset(switch):
    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)

@mF.timing_decorator
def run_pr_metrics(switch):
    print(f'Running pr metrics with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    run_dataset(switch)


# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    run_pr_metrics(switch = {
        # Type of dataset
        'constructed_fields':             False, 
        'sample_data':                    True,
        'gadi_data':                      False,

        # visuzalization
        'pr_snapshot':                    False,
        'percentiles_snapshot':           False, 
        'F_pr10_snapshot':                False,

        # metrics
        'percentiles':                    False, 
        'percentiles_sMean':              False, 
        'rxday_sMean':                    False, 
        'rxday_tMean':                    False, 
        'F_pr10':                         True,
        'o_pr':                           False,

        # threshold
        'fixed_area':                     False,

        # save3
        'save':                           True
        }
    )
    







