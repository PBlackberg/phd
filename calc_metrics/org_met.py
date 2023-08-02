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

def get_obj_snapshot(da, conv_threshold):
    ''' Snapshot to visualize the calculation by the organization metrics.
        Consists of connected components of precipiation rate exceeding threshold 
        (contiguous convective reigons) 
    '''
    scene = da.isel(time=0)
    L_obj = skm.label(scene.where(scene>=conv_threshold,0)>0, background=np.nan,connectivity=2) # label matrix first gives a unique number to all groups of connected components.
    L_obj = (L_obj>0)*1  # This line sets all unique labels to 1 
    return L_obj


def rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d):
    ''' Calculate rome for a scene: 
        Average of unique pair weight: A_a + min(1, A_b / A_d) * A_b
        where
        A_a - larger area of pair
        A_b - smaller area of pair
        A_d - shortest distance between pair boundaries (km)
    '''
    rome_allPairs = []
    shape_L = np.shape(L)
    if len(labels) ==1:
        rome_allPairs = np.sum((L==labels)*1 * aream) # ROME = area of object if singular object

    else:
        for idx, labeli in enumerate(labels[0:-1]): # compare object i and object j
            I, J = zip(*np.argwhere(L==labeli)) # find coordinates of object i
            I = list(I)
            J = list(J)
            oi_area = np.sum(np.squeeze(aream)[I,J]) # find area of object i
            Ni = len(I) # find number of gridboxes in object i
            lati3d = np.tile(lat[I],reps =[shape_L[0], shape_L[1], 1]) # replicate each gridbox lon and lat into Ni 2D slices in the shape of L
            loni3d = np.tile(lon[J],reps =[shape_L[0], shape_L[1], 1])

            if Ni > np.shape(lonm3d)[2]: # create corresponding 3D matrix from Ni copies of latm, lonm (this metrix only needs to be recreated when Ni increases from previous loop)
                lonm3d = np.tile(lonm3d[:,:,0:1],reps =[1, 1, Ni])
                latm3d = np.tile(latm3d[:,:,0:1],reps =[1, 1, Ni])

            distancei3d = mF.haversine_dist(lati3d, loni3d, latm3d[:,:,0:Ni], lonm3d[:,:,0:Ni]) # distance from gridbox in object i to every other point in the domain
            distancem = np.amin(distancei3d, axis=2) # minimum in the third dimension gives shortest distance from object i to every other point in the domain
    
            for labelj in labels[idx+1:]: # the shortest distance from object i to object j will be the minimum of the coordinates of object j in distancem

                I, J = zip(*np.argwhere(L==labelj)) # find coordinates of object j
                oj_area = np.sum(aream[I,J]) # find area of object j

                large_area = np.maximum(oi_area, oj_area) 
                small_area = np.minimum(oi_area, oj_area)
                rome_pair = large_area + np.minimum(small_area, (small_area/np.amin(distancem[I,J]))**2) # ROME of unique pair
                rome_allPairs = np.append(rome_allPairs, rome_pair)
    return np.mean(rome_allPairs)


@mF.timing_decorator
def calc_rome(da, conv_threshold):
    ''' ROME - RAdar Organization MEtric
        Define scenes of contihuous convective regions
        and puts the calculated daily values of rome in a list
    '''
    rome = []
    lat = da['lat'].data
    lon = da['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2) # used for area of object
    latm3d = np.expand_dims(latm,axis=2) # used for broadcasting
    lonm3d = np.expand_dims(lonm,axis=2)

    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        threshold = pr_day.quantile(conv_threshold, dim=('lat', 'lon'), keep_attrs=True)
        L = skm.label(pr_day.where(pr_day>=threshold,0)>0, background=0,connectivity=2)
        
        mF.connect_boundary(L)
        labels = np.unique(L)[1:] # first unique value is background
        rome = np.append(rome, rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d))
    return rome


def rome_nScene(L, labels, lat, lon, aream, latm3d, lonm3d, n, o_areaScene):
    ''' Finds n largest objects and calls rome_scene
    '''
    if len(o_areaScene) <= n:
        labels_n = labels
    else:
        labels_n = labels[o_areaScene.argsort()[-n:]]
    return rome_scene(L, labels_n, lat, lon, aream, latm3d, lonm3d)


@mF.timing_decorator
def calc_rome_n(da, conv_threshold, n): 
    ''' Define scenes of contihuous convective regions
        and finds the area of objects. 
        Then puts the calculated daily values of rome_n in a list
    '''
    rome_n = []
    lat = da['lat'].data
    lon = da['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    latm3d = np.expand_dims(latm,axis=2) # used for broadcasting
    lonm3d = np.expand_dims(lonm,axis=2)
    aream3d = np.expand_dims(aream,axis=2)

    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]

        obj3d = np.stack([(L==label) for label in labels],axis=2)*1
        o_areaScene = np.sum(obj3d * aream3d, axis=(0,1))
        rome_n = np.append(rome_n, rome_nScene(L, labels, lat, lon, aream, latm3d, lonm3d, n, o_areaScene))
    return rome_n


@mF.timing_decorator
def calc_ni(da, conv_threshold):
    ''' ni - number index
        Counts the number of object in each scene and records the area fraction convered
        by convection 
    '''
    lat = da['lat'].data
    lon = da['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)

    ni, areaf = [], []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        threshold = pr_day.quantile(conv_threshold, dim=('lat', 'lon'), keep_attrs=True)
        L = skm.label(pr_day.where(pr_day>=threshold,0)>0, background=0,connectivity=2)
        
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        o_numberScene = len(labels)

        conv_day = (pr_day.where(pr_day>=threshold,0)>0)*1
        areaf_scene = (np.sum(conv_day * aream)/np.sum(aream))*100
        
        ni = np.append(ni, o_numberScene)
        areaf = np.append(areaf, areaf_scene)
    return ni, areaf


@mF.timing_decorator
def calc_o_area(da, conv_threshold):
    ''' Area of each contiguous convective region (object) 
    '''
    lat = da['lat'].data
    lon = da['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371 # km
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    aream3d = np.expand_dims(aream,axis=2) # used for broadcasting
    o_area = []
    for day in np.arange(0,len(da.time.data)):
        pr_day = da.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_areaScene = np.sum(obj3d * aream3d, axis=(0,1))
        o_area = np.append(o_area, o_areaScene)
    return o_area


# ------------------------------------------------------------------------------------ Organize metric into dataset and save ----------------------------------------------------------------------------------------------------- #

def load_data(switch, source, dataset, options):
    da = cF.var2D if switch['constructed_fields'] else None
    if switch['sample_data']:
        folder = f'/Users/cbla0002/Documents/data/pr/sample_data/{source}'
        filename = f'{dataset}_pr_daily_{options.experiment}_{options.resolution}.nc'
        da = xr.open_dataset(folder + '/' + filename)['pr']
    da = gD.get_pr(source, dataset, options.timescale, options.experiment, options.resolution) if switch['gadi_data'] else da
    return da

def calc_metrics(switch, da, conv_threshold, source, dataset, options):
    ''' Calls metric calculation on input data and saves metric to dataset
    '''
    if switch['obj_snapshot']:
        obj_snapshot = xr.DataArray(
            data=get_obj_snapshot(da, conv_threshold), dims=['lat', 'lon'], coords={'lat': da.lat.data, 'lon': da.lon.data},
            attrs = {'Description': f'Scene of contigiuos convetive regions (objects), \
                                      with convection as precipitation rates exceeding the time-mean \
                                      {int(conv_threshold*100)}th percentile'})
        ds_obj_snapshot = xr.Dataset(data_vars = {'obj_snapshot': obj_snapshot})
        folder = f'{mV.folder_save[0]}/org/metrics/obj_snapshot/{source}'
        filename = f'{dataset}_obj_snapshot_daily_{options.experiment}_{options.resolution}.nc'
        mF.save_file(ds_obj_snapshot, folder, filename) if switch['save'] else None

    if switch['rome']:
        rome = xr.DataArray(
            data = calc_rome(da, conv_threshold), dims = ['time'], coords = {'time': da.time.data}, 
            attrs = {'units': 'km' + mF.get_super('2'),
                     'description': f'ROME based on contigiuos convetive regions (objects), \
                                    with convection as precipitation rates exceeding the \
                                    {int(conv_threshold*100)}th percentile'}) #time-mean
        ds_rome = xr.Dataset(data_vars = {'rome': rome})
        folder = f'{mV.folder_save[0]}/org/metrics/rome_equal_area/{source}'
        filename = f'{dataset}_rome_equal_area_daily_{options.experiment}_{options.resolution}.nc'
        mF.save_file(ds_rome, folder, filename) if switch['save'] else None

    if switch['rome_n']:
        n=8
        rome_n = xr.DataArray(
            data = calc_rome_n(da, conv_threshold), dims = ['time'], coords = {'time': da.time.data}, 
            attrs = {'units': 'km' + mF.get_super('2'),
                              'description': f'ROME_n based on {n} largest contigiuos convetive regions (objects) \
                               with convection as precipitation rates exceeding the time-mean \
                               {int(conv_threshold*100)}th percentile'})
        ds_rome_n = xr.Dataset(data_vars = {'rome_n': rome_n})
        folder = f'{mV.folder_save[0]}/org/metrics/rome_n/{source}'
        filename = f'{dataset}_rome_n_daily_{options.experiment}_{options.resolution}.nc'
        mF.save_file(ds_rome_n, folder, filename) if switch['save'] else None

    if switch['ni']:
        ni, areafraction = calc_ni(da, conv_threshold)
        ni = xr.DataArray(
            data=ni, 
            dims=['time'], 
            coords={'time': da.time.data},
            attrs={'units':'Nb',
                   'description': f'Number of contigiuos convetive regions (objects), \
                                    with convection as precipitation rates exceeding the time-mean \
                                    {int(conv_threshold*100)}th percentile'})
        areafraction = xr.DataArray(
            data=areafraction, 
            dims=['time'], 
            coords={'time': da.time.data},
            attrs = {'units': '%'})
        ds_numberIndex = xr.Dataset(data_vars = {'ni': ni, 'areafraction': areafraction}) 
        folder = f'{mV.folder_save[0]}/org/metrics/ni/{source}'
        filename = f'{dataset}_ni_equal_area_daily_{options.experiment}_{options.resolution}.nc'
        mF.save_file(ds_numberIndex, folder, filename) if switch['save'] else None

    if switch['o_area']:
        o_area = xr.DataArray(
            data = calc_o_area(da, conv_threshold), 
            dims = 'object',
            attrs = {'units':'km' + mF.get_super('2'),
                    'description': f'Area of contigiuos convetive regions (objects), \
                                    with convection as precipitation rates exceeding the time-mean \
                                    {int(conv_threshold*100)}th percentile'})
        ds_o_area = xr.Dataset(data_vars = {'o_area': o_area}) 
        folder = f'{mV.folder_save[0]}/org/metrics/o_area/{source}'
        filename = f'{dataset}_o_area_daily_{options.experiment}_{options.resolution}.nc'
        mF.save_file(ds_o_area, folder, filename) if switch['save'] else None

# -------------------------------------------------------------------------------- Get the data from the model / experiment and run ----------------------------------------------------------------------------------------------------- #

def run_experiment(switch, source, dataset):
    for experiment in mV.experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if mF.data_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mF.no_data(source, experiment, mF.data_exist(dataset, experiment)):
            continue
            
        options = mF.dataset_class(mV.timescales[0], experiment, mV.resolutions[0])
        da = load_data(switch, source, dataset, options)
        conv_percentile = 0.97
        conv_threshold = conv_percentile
        # conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')
        calc_metrics(switch, da, conv_threshold, source, dataset, options)

@mF.timing_decorator
def run_org_metrics(switch):
    if not switch['run']:
        return
    print(f'Running org metrics with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in mV.datasets:
        source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')
        run_experiment(switch, source, dataset)



if __name__ == '__main__':
    run_org_metrics(switch = {
        # choose data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        True,
        'gadi_data':          False,

        # choose metrics to calculate
        'obj_snapshot':       False,
        'rome':               True, 
        'rome_n':             False, 
        'ni':                 False, 
        'o_area':             False,
        
        # run/savve
        'run':                True,
        'save':               True
        }
    )
    








