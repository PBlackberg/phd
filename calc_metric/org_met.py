import numpy as np
import xarray as xr
import skimage.measure as skm

import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF                                # imports common operators
import constructed_fields as cF                     # imports fields for testing
import get_data as gD                               # imports functions to get data from gadi
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 # imports common variables
import myClasses as mC

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

def calc_conv_threshold(switch, da): # conv_percentile is number [0,1], conv_threshold is a list of thresholds
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
@mF.timing_decorator
def get_obj_snapshot(da, conv_threshold):
    ''' Connected components (objects) of convection (precipiation rate exceeding threshold) '''
    pr_day = da.isel(time=0) 
    L_obj = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=0),0)>0, background=np.nan,connectivity=2) # label matrix first gives a unique number to all groups of connected components.
    return (L_obj>0)*1  # Sets all unique labels to 1 


# ------------------------------------------------------------------------------------ Radar Organization MEtric (ROME) ----------------------------------------------------------------------------------------------------- #
def rome_scene(L, labels, dim):
    ''' Calculates ROME for a scene: 
        Average of unique pair weight: A_a + min(1, A_b / A_d) * A_b
        where
        A_a - larger area of pair
        A_b - smaller area of pair
        A_d - shortest distance between pair boundaries (km) '''
    rome_allPairs = []
    latm3d, lonm3d = dim.latm3d, dim.lonm3d # these are updated for each loop (extended in thrid dimension)
    shape_L = np.shape(L)
    if len(labels) ==1:
        rome_allPairs = np.sum((L==labels)*1 * dim.aream) # ROME = area of object if singular object
    else:
        for idx, labeli in enumerate(labels[0:-1]): # compare object i and object j
            I, J = zip(*np.argwhere(L==labeli)) # find coordinates of object i
            I, J = list(I), list(J)
            oi_area = np.sum(np.squeeze(dim.aream)[I,J]) # find area of object i
            Ni = len(I) # find number of gridboxes in object i
            lati3d = np.tile(dim.lat[I],reps =[shape_L[0], shape_L[1], 1]) # replicate each gridbox lon and lat into Ni 2D slices in the shape of L
            loni3d = np.tile(dim.lon[J],reps =[shape_L[0], shape_L[1], 1])

            if Ni > np.shape(lonm3d)[2]: # create corresponding 3D matrix from Ni copies of latm, lonm (this metrix only needs to be recreated when Ni increases from previous loop)
                lonm3d = np.tile(lonm3d[:,:,0:1],reps =[1, 1, Ni])
                latm3d = np.tile(latm3d[:,:,0:1],reps =[1, 1, Ni])

            distancei3d = mF.haversine_dist(lati3d, loni3d, latm3d[:,:,0:Ni], lonm3d[:,:,0:Ni]) # distance from gridbox in object i to every other point in the domain
            distancem = np.amin(distancei3d, axis=2) # minimum in the third dimension gives shortest distance from object i to every other point in the domain
    
            for labelj in labels[idx+1:]: # the shortest distance from object i to object j will be the minimum of the coordinates of object j in distancem

                I, J = zip(*np.argwhere(L==labelj)) # find coordinates of object j
                oj_area = np.sum(dim.aream[I,J]) # find area of object j

                large_area = np.maximum(oi_area, oj_area) 
                small_area = np.minimum(oi_area, oj_area)
                rome_pair = large_area + np.minimum(small_area, (small_area/np.amin(distancem[I,J]))**2) # ROME of unique pair
                rome_allPairs = np.append(rome_allPairs, rome_pair)
    return np.mean(rome_allPairs)

@mF.timing_decorator
def calc_rome(da, dim, conv_threshold):
    ''' ROME - RAdar Organization MEtric
        Calls rome_scene on objects '''
    rome = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        L = skm.label(pr_day.where(pr_day >= conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:] # first unique value is background
        rome = np.append(rome, rome_scene(L, labels, dim))
    return rome

def rome_nScene(L, labels, o_areaScene, n, dim):
    ''' Calls rome_scene of subset of objects '''
    if len(o_areaScene) <= n:
        labels_n = labels
    else:
        labels_n = labels[o_areaScene.argsort()[-n:]]
    return rome_scene(L, labels_n, dim)

@mF.timing_decorator
def calc_rome_n(da, dim, conv_threshold, n): 
    ''' ROME_n
        Finds n largest objects and calls rome_scene of subset of objects '''
    rome_n = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time = day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1
        o_areaScene = np.sum(obj3d * dim.aream3d, axis=(0,1))
        rome_n = np.append(rome_n, rome_nScene(L, labels, o_areaScene, n, dim))
    return rome_n


# --------------------------------------------------------------------------------------------- Number index ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def calc_ni(da, dim, conv_threshold):
    ''' Number of object in scene '''
    ni = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        o_numberScene = len(labels)
        ni = np.append(ni, o_numberScene)
    return ni


# --------------------------------------------------------------------------------------------- Areafraction ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def calc_areafraction(da, dim, conv_threshold):
    ''' Areafraction convered by convection'''
    areaf = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        conv_day = (pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0)*1
        areaf_scene = (np.sum(conv_day * dim.aream)/np.sum(dim.aream))*100
        areaf = np.append(areaf, areaf_scene)
    return areaf

# ---------------------------------------------------------------------------------------------- Object area ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def calc_o_area(da, dim, conv_threshold):
    ''' Area of each contiguous convective region (object) '''
    o_area = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_areaScene = np.sum(obj3d * dim.aream3d, axis=(0,1))
        o_area = np.append(o_area, o_areaScene)
    return o_area

@mF.timing_decorator
def calc_mean_area(da, dim, conv_threshold):
    ''' Mean area of each contiguous convective region (object) in a day'''
    mean_area = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold.isel(time=day),0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_areaScene = np.sum(obj3d * dim.aream3d, axis=(0,1))
        mean_area = np.append(mean_area, np.sum(o_areaScene)/len(labels))
    return mean_area


# -------------------------------------------------------------------------------------------------- Other ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def calc_F_pr10(da):
    ''' Frequency of gridboxes exceeding 10 mm/day on monthly [% of domain]'''
    da = mF.resample_timeMean(da, 'M')
    mask = xr.where(da>10,1,0)
    F_pr10 = (mask.sum(dim=('lat','lon')) / (len(da['lat']) * len(da['lon']))) * 100
    return F_pr10



# ------------------------
#   Run / save metrics
# ------------------------
# -------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, source, dataset, experiment, da, dim, conv_threshold, metric):
    ''' Calls metric calculation on input data and puts metric in dataset '''
    da_calc, metric_name = None, None
    if metric == 'obj_snapshot':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        da_calc = xr.DataArray(data = get_obj_snapshot(da, conv_threshold), dims=['lat', 'lon'], coords={'lat': da.lat.data, 'lon': da.lon.data},
                               attrs = {'description': f'Contigiuos convetive regions (objects). Threshold: {mV.conv_percentiles[0]}th percentile'})
    if metric == 'rome':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        da_calc = xr.DataArray(data = calc_rome(da, dim, conv_threshold), dims = ['time'], coords = {'time': da.time.data}, 
                               attrs = {'units': r'km$^2$', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})

    if metric == 'rome_n':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        n=8
        da_calc = xr.DataArray(data = calc_rome_n(da, dim, conv_threshold, n), dims = ['time'], coords = {'time': da.time.data}, 
                               attrs = {'units': r'km$^2$', 'description': f'{n} largest contigiuos convetive regions (objects). Threshold: {mV.conv_percentiles[0]}th percentile'})

    if metric == 'ni':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        da_calc = xr.DataArray(data = calc_ni(da, dim, conv_threshold), dims=['time'], coords={'time': da.time.data},
                               attrs = {'units':'Nb', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})
        
    if metric == 'areafraction':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        da_calc = xr.DataArray(data = calc_areafraction(da, dim, conv_threshold), dims=['time'], coords={'time': da.time.data},
                                    attrs = {'units': '%', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})

    if metric == 'o_area':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        da_calc = xr.DataArray(data = calc_o_area(da, dim, conv_threshold), dims = 'obj', 
                               attrs = {'units':r'km$^2$', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})
    
    if metric == 'mean_area':
        metric_name = f'{metric}_{mV.conv_percentiles[0]}thprctile'
        da_calc = xr.DataArray(data = calc_mean_area(da, dim, conv_threshold), dims = ['time'], coords = {'time': da.time.data}, 
                               attrs = {'units': r'km$^2$', 'description': f'Threshold: {mV.conv_percentiles[0]}th percentile'})
        
    metric_name = f'{metric_name}_fixed_area' if switch['fixed_area'] else metric_name

    if metric == 'F_pr10':
        metric_name = metric
        da_calc = calc_F_pr10(da)

    mF.save_in_structured_folders(da_calc, f'{mV.folder_save[0]}/metrics', 'org', metric_name, source, dataset, mV.timescales[0], experiment, mV.resolutions[0])                                   if (switch['save'] and da_calc is not None)            else None
    mF.save_file(xr.Dataset(data_vars = {metric_name: da_calc}), f'{home}/Desktop/{metric_name}', f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc') if (switch['save_to_desktop'] and da_calc is not None) else None



# ---------------------------------------------------------------------------------------------------- pick dataset ----------------------------------------------------------------------------------------------------- #
def run_metrics(switch, source, dataset, experiment, da, dim, conv_threshold):
    for metric in [k for k, v in switch.items() if v] : # loop over true keys
        if metric in ['obj_snapshot', 'rome', 'rome_n', 'ni', 'areafraction', 'o_area', 'mean_area', 'F_pr10']:
            get_metric(switch, source, dataset, experiment, da, dim, conv_threshold, metric)

@mF.timing_decorator
def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment):
            continue
        print(f'\t {experiment}') if experiment else print(f'\t observational dataset')
        da = load_data(switch, source, dataset, experiment)
        dim = mC.dims_class(da)
        conv_threshold = calc_conv_threshold(switch, da)
        run_metrics(switch, source, dataset, experiment, da, dim, conv_threshold)

@mF.timing_decorator
def run_dataset(switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)

@mF.timing_decorator
def run_org_metrics(switch):
    print(f'Running {os.path.basename(__file__)} on {mV.resolutions[0]} {mV.timescales[0]} data with {mV.conv_percentiles[0]}th percentile precipitation threshold')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    run_dataset(switch)



# ------------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    run_org_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False,
        'sample_data':        True,
        'gadi_data':          False,

        # threshold
        'fixed_area':         False,

        # choose metric
        'obj_snapshot':       True,
        'rome':               True, 
        'rome_n':             True, 
        'ni':                 True, 
        'areafraction':       True, 
        'o_area':             True,
        'mean_area':          True,
        'F_pr10':             True,

        # save
        'save':               True,
        'save_to_desktop':    False
        }
    )
    









