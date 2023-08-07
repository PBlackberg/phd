import numpy as np
import xarray as xr
import skimage.measure as skm
import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD # imports functions to get data from gadi


# ----------------------------------------------------------------------------------------------- Calculattion ----------------------------------------------------------------------------------------------------- #

def calc_rxday_sMean(da):
    ''' Most extreme daily gridpoint value locally over set time period '''
    aWeights = np.cos(np.deg2rad(da.lat))
    rx1day = da.resample(time='Y').max(dim='time')
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    return rx1day.weighted(aWeights).mean(dim=('lat', 'lon')), rx5day.weighted(aWeights).mean(dim=('lat', 'lon'))

def calc_rxday_tMean(da):
    ''' Time-mean of local extremes over set time period '''
    rx1day = da.resample(time='Y').max(dim='time')
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    return rx1day.mean(dim='time'), rx5day.mean(dim='time')

def find_percentile(da, percentile):
    ''' Spatial percentile of the scene '''
    return da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)

def calc_percentile_snapshot(da, percentile):
    ''' snapshot of gridboxes exceeding percentile threshold '''
    da_snapshot = da.isel(time=0)
    return da_snapshot.where(da_snapshot>=find_percentile(da, percentile).isel(time=0))

def calc_meanInPercentile(da, percentile):
    ''' Mean precipitation rate of the gridboxes included in the percentile of each scene (precipiration rate threshold) '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.where(da >= find_percentile(da, percentile).mean(dim='time')).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)

def calc_meanInPercentile_fixedArea(da, percentile):
    ''' Mean precipitation rate of the gridboxes included in the percentile of each scene (fixed area threshold) '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.where(da >= find_percentile(da, percentile)).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)

def calc_F_pr10(da):
    ''' Frequency of gridboxes exceeding 10 mm/day on monthly'''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    return mask.sum(dim=('lat','lon'))

def calc_F_pr10_snapshot(da):
    ''' Snapshot of gridboxes exceeding 10 mm/day '''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    return da.isel(time=0).where(mask.isel(time=0) > 0)

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
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_prScene = np.sum(obj3d * pr_day3d * aream3d, axis=(0,1)) / np.sum(obj3d * aream3d, axis=(0,1))
        o_pr = np.append(o_pr, o_prScene)
    return o_pr


# ---------------------------------------------------------------------------------------------- Put in dataset and save ----------------------------------------------------------------------------------------------------- #
    
def calc_metrics(switch, da, source, dataset, experiment):
    if switch['snapshot_pr']:
        ds_pr_snapshot = xr.Dataset({f'snapshot_pr' : da.isel(time=0)})
        folder = f'{mV.folder_save[0]}/pr/metrics/snapshot_pr/{source}'
        filename = f'{dataset}_snapshot_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_pr_snapshot, folder, filename) if switch['save'] else None

    if switch['rxday_sMean']:
        rx1day_sMean, rx5day_sMean = calc_rxday_sMean(da)
        ds_rxday_sMean = xr.Dataset({'rx1day_sMean_pr': rx1day_sMean , 'rx5day_sMean_pr': rx5day_sMean})         
        folder = f'{mV.folder_save[0]}/pr/metrics/rxday_sMean_pr/{source}'
        filename = f'{dataset}_rxday_sMean_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_rxday_sMean, folder, filename) if switch['save'] else None

    if switch['rxday_tMean']:
        rx1day_tMean, rx5day_tMean = calc_rxday_tMean(da)
        ds_rxday_tMean = xr.Dataset({'rx1day_tMean_pr': rx1day_tMean , 'rx5day_tMean_pr': rx5day_tMean})
        folder = f'{mV.folder_save[0]}/pr/metrics/rxday_tMean_pr/{source}'
        filename = f'{dataset}_rxday_tMean_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_rxday_tMean, folder, filename) if switch['save'] else None

    if switch['percentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_percentile_value = xr.Dataset()
        for percentile in percentiles:
            ds_percentile_value[f'pr{int(percentile*100)}'] = find_percentile(da, percentile)
        folder = f'{mV.folder_save[0]}/pr/metrics/percentiles_pr/{source}'
        filename = f'{dataset}_percentiles_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_percentile_value, folder, filename) if switch['save'] else None
        
    if switch['snapshot_percentiles']:
        ds_percentile_snapshot = xr.Dataset()
        for percentile in percentiles:
            percentile_snapshot= calc_percentile_snapshot(da, percentile)
            ds_percentile_snapshot[f'snapshot_pr{int(percentile*100)}'] = percentile_snapshot
        folder = f'{mV.folder_save[0]}/pr/metrics/percentiles_pr_snapshot/{source}'
        filename = f'{dataset}_percentiles_pr_snapshot_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_percentile_value, folder, filename) if switch['save'] else None

    if switch['meanInPercentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_meanInPercentiles = xr.Dataset()
        for percentile in percentiles:
            meanInPercentile = calc_meanInPercentile(da, percentile)
            ds_meanInPercentiles[f'pr{int(percentile*100)}_meanIn'] = meanInPercentile
        folder = f'{mV.folder_save[0]}/pr/metrics/meanInPercentiles_pr/{source}'
        filename = f'{dataset}_meanInPercentiles_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_meanInPercentiles, folder, filename) if switch['save'] else None

    if switch['meanInPercentiles_fixedArea']:
        percentiles = [0.95, 0.97, 0.99]
        ds_meanInPercentiles = xr.Dataset()
        for percentile in percentiles:
            meanInPercentile = calc_meanInPercentile_fixedArea(da, percentile)
            ds_meanInPercentiles[f'pr{int(percentile*100)}_meanIn_fixedArea'] = meanInPercentile
        folder = f'{mV.folder_save[0]}/pr/metrics/meanInPercentiles_fixedArea_pr/{source}'
        filename = f'{dataset}_meanInPercentiles_fixedArea_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_meanInPercentiles, folder, filename) if switch['save'] else None

    if switch['snapshot_F_pr10']:
        ds_F_pr10_snapshot = xr.Dataset({'F_pr10': calc_F_pr10_snapshot(da)})
        folder = f'{mV.folder_save[0]}/pr/metrics/F_pr10_snapshot/{source}'
        filename = f'{dataset}_F_pr10_snapshot_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_F_pr10_snapshot, folder, filename) if switch['save'] else None

    if switch['F_pr10']:
        ds_F_pr10 = xr.Dataset({'F_pr10': calc_F_pr10(da)})
        folder = f'{mV.folder_save[0]}/pr/metrics/F_pr10/{source}'
        filename = f'{dataset}_F_pr10_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_F_pr10, folder, filename) if switch['save'] else None

    if switch['o_pr']:
        conv_percentile = 0.97
        conv_threshold = find_percentile(da, conv_percentile).mean(dim='time')
        o_pr = xr.DataArray(data = calc_o_pr(da, conv_threshold), dims = 'object',
                                attrs = {'units':'mm day' + mF.get_super('-1'),
                                         'descrption': 'area weighted mean precipitation in contiguous convective region (object)'})
        ds_o_pr = xr.Dataset({'o_pr': o_pr})
        folder = f'{mV.folder_save[0]}/pr/metrics/o_pr/{source}'
        filename = f'{dataset}_o_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_o_pr, folder, filename) if switch['save'] else None


# -------------------------------------------------------------------------------------------- Load data ----------------------------------------------------------------------------------------------------- #

def load_data(switch, source, dataset, experiment):
    if switch['constructed_fields']:
        return cF.var2D
    if switch['sample_data']:
        path = f'/Users/cbla0002/Documents/data/pr/sample_data/{source}/{dataset}_pr_daily_{experiment}_{mV.resolutions[0]}.nc'
        return xr.open_dataset(path)['pr']
    else:
        return gD.get_pr(source, dataset, mV.timescales[0], experiment, mV.resolutions[0])

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mF.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None
        if mF.no_data(source, experiment, mF.data_exist(dataset, experiment)):
            continue
        da = load_data(switch, source, dataset, experiment)
        calc_metrics(switch, da, source, dataset, experiment)

@mF.timing_decorator
def run_pr_metrics(switch):
    print(f'Running pr metrics with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)



# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':
    run_pr_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,

        # choose metrics to calculate
        'snapshot_pr':                    False,
        'rxday_sMean':                    False, 
        'rxday_tMean':                    False, 
        'percentiles':                    False, 
        'snapshot_percentiles':           False, 
        'meanInPercentiles':              False, 
        'meanInPercentiles_fixedArea':    False,
        'F_pr10':                         False,
        'snapshot_F_pr10':                False,
        'o_pr':                           False,
        
        # save
        'save':               False
        }
    )
    













