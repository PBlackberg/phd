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


# ------------------------------------------------------------------------------------- Calculating metric from data array ----------------------------------------------------------------------------------------------------- #

def calc_rx1day(da):
    ''' Most extreme daily gridpoint value locally over time (1 year here)
    '''
    rx1day = da.resample(time='Y').max(dim='time')
    aWeights = np.cos(np.deg2rad(da.lat))
    rx1day_tMean = rx1day.mean(dim='time')
    rx1day_sMean = rx1day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx1day_tMean, rx1day_sMean

def calc_rx5day(da):
    ''' Most extreme 5-day average gridpoint value locally over time (1 year here)
    '''
    da5day = da.resample(time='5D').mean(dim='time')
    rx5day = da5day.resample(time='Y').max(dim='time')
    aWeights = np.cos(np.deg2rad(da.lat))
    rx5day_tMean = rx5day.mean(dim='time')
    rx5day_sMean = rx5day.weighted(aWeights).mean(dim=('lat', 'lon'))
    return rx5day_tMean, rx5day_sMean

def find_percentile(da, percentile):
    ''' Spatial percentile of the scene
    '''
    percentile_value = da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)
    return percentile_value

def calc_percentile_snapshot(da, percentile):
    ''' snapshot of gridboxes exceeding percentile threshold
    '''
    da_snapshot = da.isel(time=0)
    return da_snapshot.where(da_snapshot>=find_percentile(da, percentile).isel(time=0))

def calc_meanInPercentile(da, percentile):
    ''' Mean precipitation rate of the gridboxes included in the percentile of each scene
    '''
    aWeights = np.cos(np.deg2rad(da.lat))
    meanInPercentile = da.where(da >= find_percentile(da, percentile)).weighted(aWeights).mean(dim=('lat', 'lon'), keep_attrs=True)
    return meanInPercentile

def calc_F_pr10(da):
    ''' Frequency of gridboxes exceeding 10 mm/day
    '''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    F_pr10 = mask.sum(dim=('lat','lon'))
    return F_pr10

def calc_F_pr10_snapshot(da):
    ''' snapshot of frequency of gridboxes exceeding 10 mm/day
    '''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    return da.isel(time=0).where(mask.isel(time=0) > 0)

def calc_o_pr(da, conv_threshold):
    ''' Precipitation rate in each contigous convective region (object)
    '''
    lat = da['lat'].data
    lon = da['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
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


# ------------------------------------------------------------------------------------ Organize metric into dataset and save ----------------------------------------------------------------------------------------------------- #

def load_data(switch, source, dataset, experiment):
    da = cF.var2D if switch['constructed_fields'] else None
    if switch['sample_data']:
        folder = f'/Users/cbla0002/Documents/data/pr/sample_data/{source}'
        filename = f'{dataset}_pr_daily_{experiment}_{mV.resolutions[0]}.nc'
        da = xr.open_dataset(folder + '/' + filename)['pr']
    da = gD.get_pr(source, dataset, mV.timescales[0], experiment, mV.resolutions[0]) if switch['gadi_data'] else da
    return da
    
def calc_metrics(switch, da, source, dataset, experiment):
    if switch['pr_snapshot']:
        ds_pr_snapshot = xr.Dataset({f'pr_snapshot' : da.isel(time=0)})

        folder = f'{mV.folder_save[0]}/pr/metrics/pr_snapshot/{source}'
        filename = f'{dataset}_pr_snapshot_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_pr_snapshot, folder, filename) if switch['save'] else None

    if switch['rxday_pr']:
        rx1day_tMean, rx1day_sMean = calc_rx1day(da)
        rx5day_tMean, rx5day_sMean = calc_rx5day(da)
        ds_rxday_tMean = xr.Dataset({'rx1day_pr_tMean': rx1day_tMean , 'rx5day_pr_tMean': rx5day_tMean})

        folder = f'{mV.folder_save[0]}/pr/metrics/rxday_pr_tMean/{source}'
        filename = f'{dataset}_rxday_pr_tMean_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_rxday_tMean, folder, filename) if switch['save'] else None

        ds_rxday_sMean = xr.Dataset({'rx1day_pr_sMean': rx1day_sMean , 'rx5day_pr_sMean': rx5day_sMean}) 
        folder = f'{mV.folder_save[0]}/pr/metrics/rxday_pr_sMean/{source}'
        filename = f'{dataset}_rxday_pr_sMean_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_rxday_sMean, folder, filename) if switch['save'] else None

    if switch['percentiles_snapshot']:
        ds_percentile_snapshot = xr.Dataset()
        for percentile in percentiles:
            percentile_snapshot= calc_percentile_snapshot(da, percentile)
            ds_percentile_snapshot[f'pr{int(percentile*100)}_snapshot'] = percentile_snapshot

        folder = f'{mV.folder_save[0]}/pr/metrics/percentiles_pr_snapshot/{source}'
        filename = f'{dataset}_percentiles_pr_snapshot_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_percentile_value, folder, filename) if switch['save'] else None

    if switch['percentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_percentile_value = xr.Dataset()
        for percentile in percentiles:
            ds_percentile_value[f'pr{int(percentile*100)}'] = find_percentile(da, percentile)

        folder = f'{mV.folder_save[0]}/pr/metrics/percentiles_pr/{source}'
        filename = f'{dataset}_percentiles_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_percentile_value, folder, filename) if switch['save'] else None
        
    if switch['meanInPercentiles']:
        percentiles = [0.95, 0.97, 0.99]
        ds_meanInPercentiles = xr.Dataset()
        for percentile in percentiles:
            meanInPercentile = calc_meanInPercentile(da, percentile)
            ds_meanInPercentiles[f'pr{int(percentile*100)}'] = meanInPercentile

        folder = f'{mV.folder_save[0]}/pr/metrics/meanInPercentiles_pr/{source}'
        filename = f'{dataset}_meanInPercentiles_pr_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}'
        mF.save_file(ds_meanInPercentiles, folder, filename) if switch['save'] else None

    if switch['F_pr10_snapshot']:
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

# -------------------------------------------------------------------------------- Get the data from the dataset / experiment and run ----------------------------------------------------------------------------------------------------- #

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
    if not switch['run']:
        return
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
        'gadi_data':          False,

        # choose metrics to calculate
        'pr_snapshot':          True,
        'rxday_pr':             False, 
        'percentiles_snapshot': False, 
        'percentiles':          False, 
        'meanInPercentiles':    False, 
        'F_pr10_snapshot':      False,
        'F_pr10':               False,
        'o_pr':                 False,
        
        # run/savve
        'run':                True,
        'save':               True
        }
    )
    













