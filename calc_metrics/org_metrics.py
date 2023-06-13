import numpy as np
import xarray as xr
import timeit
import skimage.measure as skm

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data.pr_data as pD # imports functions to get data from gadi




# ------------------------------------------------------------------------------------- Calculating metric from data array ----------------------------------------------------------------------------------------------------- #

def get_o_scene(da, conv_threshold):
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
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
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
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        mF.connect_boundary(L)
        labels = np.unique(L)[1:]
        o_numberScene = len(labels)

        conv_day = (pr_day.where(pr_day>=conv_threshold,0)>0)*1
        areaf_scene = (np.sum(conv_day * aream)/np.sum(aream))*100
        
        ni = np.append(ni, o_numberScene)
        areaf = np.append(areaf, areaf_scene)
    return ni, areaf


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

def calc_metrics(switch, da, folder_save, source, dataset, experiment, conv_threshold):
    ''' Calls metric calculation on input data and saves metric to dataset
    '''
    if switch['o_scene']:
        o_scene = xr.DataArray(data=get_o_scene(da, conv_threshold), dims=['lat', 'lon'], coords={'lat': da.lat.data, 'lon': da.lon.data})
        ds_o_scene = xr.Dataset({'o_scene':o_scene})
        mV.save_metric(ds_o_scene, folder_save, 'o_scene', source, dataset, experiment) if switch['save'] else None

    if switch['rome']:
        ds_rome = xr.DataArray(data = calc_rome(da, conv_threshold), dims = ['time'], coords = {'time': da.time.data}, 
                               attrs = {'units':'km' + mF.get_super('2'),
                                        'Description': f'ROME based on convection from {int(conv_threshold*100)}th percentile precipitation'})
        
        mV.save_metric(ds_rome, folder_save, 'rome', source, dataset, experiment) if switch['save'] else None
    if switch['rome_n']:
        n=8
        ds_rome = xr.DataArray(data = calc_rome_n(da, conv_threshold), dims = ['time'], coords = {'time': da.time.data}, 
                               attrs = {'units':'km' + mF.get_super('2'),
                                        'Description': f'ROME_n based on {n} largest contigiuos convetive regions (objects) \
                                        from {int(conv_threshold*100)}th percentile precipitation'})
        mV.save_metric(ds_rome, folder_save, 'rome_n', source, dataset, experiment) if switch['save'] else None

    if switch['ni']:
        ni, areafraction = calc_ni(da, conv_threshold)
        ni = xr.DataArray(data=ni, dims=['time'], coords={'time': da.time.data},
                                attrs={'units':'Nb',
                                       'Description': f'Number of contigiuos convetive regions (objects) in convection \
                                        from {int(conv_threshold*100)}th percentile precipitation'})
        areafraction = xr.DataArray(data=areafraction, dims=['time'], coords={'time': da.time.data},
                                attrs = {'units': '%'})
        ds_numberIndex = xr.Dataset(data_vars = {'ni': ni, 'areafraction': areafraction}) 
        mV.save_metric(ds_numberIndex, folder_save, 'ni', source, dataset, experiment) if switch['save'] else None

    if switch['o_area']:
        o_area = calc_oArea(da, conv_threshold)
        ds_o_area = xr.DataArray(data = o_area, dims = 'object',
                                attrs = {'units':'km' + mF.get_super('2')})
                
        mV.save_metric(ds_o_area, folder_save, 'o_area', source, dataset, experiment) if switch['save'] else None


# -------------------------------------------------------------------------------- Get the data from the model / experiment and run ----------------------------------------------------------------------------------------------------- #

def load_data(switch, dataset, experiment, folder_save, timescale, resolution):
    if switch['constructed_fields']:
        return cF.var2D
    elif switch['sample_data']:
        return mV.load_sample_data(f'{mV.folder_save}/pr', dataset, 'pr', timescale, experiment, resolution)['precip']
    else:
        return pD.get_pr('pr', dataset, experiment, timescale, resolution)['pr']


def run_experiment(switch, source, dataset, experiments, folder_save, timescale, resolution):
    for experiment in experiments:
        if experiment and source in ['cmip5', 'cmip6']:
            print(f'\t {experiment}') if pD.prData_exist(dataset, experiment) else print(f'\t no {experiment} data')
        print( '\t obserational dataset') if not experiment and source == 'obs' else None

        if mV.no_data(source, experiment, pD.prData_exist(dataset, experiment)):
            continue

        da = load_data(switch, dataset, experiment, folder_save, timescale, resolution)
        conv_percentile = 0.97
        conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')
        calc_metrics(switch, da, folder_save, source, dataset, experiment, conv_threshold)


def run_org_metrics(switch, datasets, experiments, timescale = 'daily', resolution= 'regridded', folder_save = f'{mV.folder_save}/org'):
    print(f'Running org metrics with {resolution} {timescale} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    for dataset in datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'{dataset} ({source})')

        run_experiment(switch, source, dataset, experiments, folder_save, timescale, resolution)





if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to calculate
    switch = {
        'constructed_fields': False, 
        'sample_data': True,

        'o_scene':False,
        'rome': False, 
        'rome_n': False, 
        'ni': True, 
        'o_area': False,
        
        'save': True
        }

    # choose which datasets and experiments to run, and where to save the metric
    ds_metric = run_org_metrics(switch=switch,
                                   datasets = mV.datasets, 
                                   experiments = mV.experiments,
                                #    folder_save = f'{mV.folder_save_gadi}/org'
                                   )
    

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
    

















# if __name__ == '__main__':
   
#     for dataset in datasets:
#         print(dataset)
#         start = timeit.default_timer()

#         for experiment in experiments:
#             if not data_exist(dataset, experiment):
#                 print(f'no {experiment} data')
#             else:
#                 print(experiment)

#                 # load data
#                 if run_on_gadi:
#                     if dataset == 'GPCP':
#                         from obs_variables import *
#                         precip = get_GPCP(institutes[model], model, experiment)['precip']
                    
#                     if np.isin(models_cmip5, dataset).any():
#                         from cmip5_variables import *
#                         precip = get_pr(institutes[model], model, experiment)['precip']
                    
#                     if run_on_gadi and np.isin(models_cmip6, dataset).any():
#                         from cmip6_variables import *
#                         precip = get_pr(institutes[model], model, experiment)['precip']
#                 else:
#                     precip = get_dsvariable('precip', dataset, experiment, home, resolutions[0])['precip']
                

#                 # Calculate diagnostics and put into dataset
#                 quantile_threshold = 0.97
#                 conv_threshold = precip.quantile(quantile_threshold,dim=('lat','lon'),keep_attrs=True).mean(dim='time',keep_attrs=True)
#                 n = 8
#                 rome = calc_rome(precip, conv_threshold)
#                 # rome_n = calc_rome_n(n, precip, conv_threshold)
#                 ds_rome = xr.Dataset(
#                     data_vars = {'rome':rome, 
#                                 'rome_n':np.nan},
#                     attrs = {'description': 'ROME based on all and the {} largest objects in the scene for each day'.format(n)}                  
#                     )
#                 ds_numberIndex = calc_numberIndex(precip, conv_threshold)
#                 ds_oAreaAndPr = calc_oAreaAndPr(precip, conv_threshold)



#                 # save
#                 save_rome = True
#                 save_numberIndex = True
#                 save_oAreaAndPr = True
                
#                 if np.isin(models_cmip5, dataset).any():
#                     project = 'cmip5'
#                 elif np.isin(models_cmip6, dataset).any():
#                     project = 'cmip6'
#                 elif np.isin(observations, dataset).any():
#                     project = 'obs'
#                 folder_save = home + '/data/' + project + '/' + 'metrics_' + project + '_' + resolutions[0] + '/' + dataset 

#                 if save_rome:
#                     fileName = dataset + '_rome_' + experiment + '_' + resolutions[0] + '.nc'              
#                     save_file(ds_rome, folder_save, fileName)

#                 if save_numberIndex:
#                     fileName = dataset + '_numberIndex_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_numberIndex, folder_save, fileName) 

#                 if save_oAreaAndPr:
#                     fileName = dataset + '_oAreaAndPr_' + experiment + '_' + resolutions[0] + '.nc'
#                     save_file(ds_oAreaAndPr, folder_save, fileName)


#             stop = timeit.default_timer()
#             print('dataset: {} took {} minutes to finsih'.format(dataset, (stop-start)/60))











