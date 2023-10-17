import numpy as np
import xarray as xr
import skimage.measure as skm


import timeit
import os
import sys
run_on_gadi = False
if run_on_gadi:
    home = '/g/data/k10/cb4968'
    sys.path.insert(0, '{}/phd/metrics/get_variables'.format(home))
else:
    home = os.path.expanduser("~") + '/Documents'
sys.path.insert(0, '{}/phd/functions'.format(home))
from myFuncs import *
# import constructed_fields as cf


def connect_boundary(array):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) 
    '''
    s = np.shape(array)
    for row in np.arange(0,s[0]):
        if array[row,0]>0 and array[row,-1]>0:
            array[array==array[row,0]] = min(array[row,0],array[row,-1])
            array[array==array[row,-1]] = min(array[row,0],array[row,-1])


def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula) 
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 

    where 
    phi -latitutde
    lambda - longitude
    (Takes vectorized input)
    '''
    R = 6371 # radius of earth in km
    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180) # function requires lon [-180 to 180]
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)
    
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2 # Haversine formula
    return 2 * R * np.arcsin(np.sqrt(h))


def calc_rome(precip, conv_threshold):
    ''' Define scenes of contihuous convective regions
    and puts the calculated daily values of rome in a list
    '''
    rome = []
    lat = precip['lat'].data
    lon = precip['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2) # used for area of object
    latm3d = np.expand_dims(latm,axis=2) # used for broadcasting
    lonm3d = np.expand_dims(lonm,axis=2)

    for day in np.arange(0,len(precip.time.data)):
        pr_day = precip.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        connect_boundary(L)
        labels = np.unique(L)[1:] # first unique value is background
        rome = np.append(rome, rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d))
    
    rome = xr.DataArray(
        data = rome,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'km\u00b2'}
        )
    return rome


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

            distancei3d = haversine_dist(lati3d, loni3d, latm3d[:,:,0:Ni], lonm3d[:,:,0:Ni]) # distance from gridbox in object i to every other point in the domain
            distancem = np.amin(distancei3d, axis=2) # minimum in the third dimension gives shortest distance from object i to every other point in the domain
    
            for labelj in labels[idx+1:]: # the shortest distance from object i to object j will be the minimum of the coordinates of object j in distancem

                I, J = zip(*np.argwhere(L==labelj)) # find coordinates of object j
                oj_area = np.sum(aream[I,J]) # find area of object j

                large_area = np.maximum(oi_area, oj_area) 
                small_area = np.minimum(oi_area, oj_area)
                rome_pair = large_area + np.minimum(small_area, (small_area/np.amin(distancem[I,J]))**2) # ROME of unique pair
                rome_allPairs = np.append(rome_allPairs, rome_pair)
    return np.mean(rome_allPairs)


def calc_rome_n(n, precip, conv_threshold): 
    ''' Define scenes of contihuous convective regions
    and find the area of objects. 
    Then puts the calculated daily values of rome_n in a list
    '''
    rome_n = []
    lat = precip['lat'].data
    lon = precip['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    latm3d = np.expand_dims(latm,axis=2) # used for broadcasting
    lonm3d = np.expand_dims(lonm,axis=2)
    aream3d = np.expand_dims(aream,axis=2)

    for day in np.arange(0,len(precip.time.data)):
        pr_day = precip.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        connect_boundary(L)
        labels = np.unique(L)[1:]

        obj3d = np.stack([(L==label) for label in labels],axis=2)*1
        o_areaScene = np.sum(obj3d * aream3d, axis=(0,1))
        rome_n = np.append(rome_n, rome_nScene(n, o_areaScene, L, labels, lat, lon, aream, latm3d, lonm3d))
    
    rome_n = xr.DataArray(
        data = rome_n,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'description':'rome calculated from {} largest contigiuos convetive areas'.format(n),
                 'units': 'km\u00b2'}
        )
    return rome_n

def rome_nScene(n, o_areaScene, L, labels, lat, lon, aream, latm3d, lonm3d):
    ''' Finds n largest objects and calls rome_scene'''
    if len(o_areaScene) <= n:
        labels_n = labels
    else:
        labels_n = labels[o_areaScene.argsort()[-n:]]
    return rome_scene(L, labels_n, lat, lon, aream, latm3d, lonm3d)


def calc_numberIndex(precip, conv_threshold):
    ''' Counts the number of object in each scene and records the area fraction convered
    by convection'''
    lat = precip['lat'].data
    lon = precip['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)

    o_number, areaf = [], []
    for day in np.arange(0,len(precip.time.data)):
        pr_day = precip.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        connect_boundary(L)
        labels = np.unique(L)[1:]
        o_numberScene = len(labels)

        conv_day = (pr_day.where(pr_day>=conv_threshold,0)>0)*1
        areaf_scene = np.sum(conv_day * aream)/np.sum(aream)
        
        o_number = np.append(o_number, o_numberScene)
        areaf = np.append(areaf, areaf_scene)

    o_number = xr.DataArray(
        data=o_number,
        dims=['time'],
        coords={'time': precip.time.data},
        attrs={'units':'Nb'}
        )

    areaf = xr.DataArray(
        data=areaf*100,
        dims=['time'],
        coords={'time': precip.time.data},
        attrs = {'units': '%'}
        )

    ds_numberIndex = xr.Dataset(
        data_vars = {'o_number': o_number, 
                     'areaf': areaf}
        ) 
    return ds_numberIndex


def calc_oAreaAndPr(precip, conv_threshold):
    '''Area and precipitation rate of each object 
    '''
    lat = precip['lat'].data
    lon = precip['lon'].data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371 # km
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    aream3d = np.expand_dims(aream,axis=2) # used for broadcasting

    o_pr, o_area = [], []
    for day in np.arange(0,len(precip.time.data)):
        pr_day = precip.isel(time=day)
        pr_day3d = np.expand_dims(pr_day,axis=2)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        connect_boundary(L)
        labels = np.unique(L)[1:]
        
        obj3d = np.stack([(L==label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        
        o_areaScene = np.sum(obj3d * aream3d, axis=(0,1))
        o_prScene = np.sum(obj3d * pr_day3d * aream3d, axis=(0,1)) / np.sum(obj3d * aream3d, axis=(0,1))
        
        o_area = np.append(o_area, o_areaScene)
        o_pr = np.append(o_pr, o_prScene)
        
    o_area = xr.DataArray(
        data = o_area,
        dims = 'region',
        attrs = {'units':'km\u00b2'}
        )

    o_pr = xr.DataArray(
        data = o_pr,
        dims = 'region',
        attrs = {'descrption': 'area weighted mean precipitation in contiguous convective region',
                'units':'mm day' + chr(0x207B) + chr(0x00B9)}
        )
    
    ds_oAreaAndPr = xr.Dataset(
        data_vars = {'o_area': o_area,
                    'o_pr': o_pr},
        attrs = {'descrption': 'area and precipipitation rate of contiguous convective regions in scene'}
        )
    return ds_oAreaAndPr


if __name__ == '__main__':
   
    models_cmip5 = [
        # 'IPSL-CM5A-MR', # 1
        # 'GFDL-CM3',     # 2
        # 'GISS-E2-H',    # 3
        # 'bcc-csm1-1',   # 4
        # 'CNRM-CM5',     # 5
        # 'CCSM4',        # 6
        # 'HadGEM2-AO',   # 7
        # 'BNU-ESM',      # 8
        # 'EC-EARTH',     # 9
        # 'FGOALS-g2',    # 10
        # 'MPI-ESM-MR',   # 11
        # 'CMCC-CM',      # 12
        # 'inmcm4',       # 13
        # 'NorESM1-M',    # 14
        # 'CanESM2',      # 15
        # 'MIROC5',       # 16
        # 'HadGEM2-CC',   # 17
        # 'MRI-CGCM3',    # 18
        # 'CESM1-BGC'     # 19
        ]
    
    models_cmip6 = [
        # 'TaiESM1',        # 1
        # 'BCC-CSM2-MR',    # 2
        # 'FGOALS-g3',      # 3
        # 'CNRM-CM6-1',     # 4
        # 'MIROC6',         # 5
        # 'MPI-ESM1-2-HR',  # 6
        # 'NorESM2-MM',     # 7
        # 'GFDL-CM4',       # 8
        # 'CanESM5',        # 9
        # 'CMCC-ESM2',      # 10
        # 'UKESM1-0-LL',    # 11
        # 'MRI-ESM2-0',     # 12
        # 'CESM2',          # 13
        # 'NESM3'           # 14
        ]
    
    observations = [
        # 'GPCP'
        ]
    
    datasets = models_cmip5 + models_cmip6 + observations

    resolutions = [
        # 'orig',
        'regridded'
        ]

    experiments = [
        'historical',
        # 'rcp85',
        # 'abrupt-4xCO2'
        # ''
        ]

    for dataset in datasets:
        print(dataset, 'started')
        start = timeit.default_timer()

        for experiment in experiments:
            print(experiment, 'started') 

            # load data
            if run_on_gadi:
                if dataset == 'GPCP':
                    from obs_variables import *
                    precip = get_GPCP(institutes[model], model, experiment)['precip']
                
                if np.isin(models_cmip5, dataset).any():
                    from cmip5_variables import *
                    precip = get_pr(institutes[model], model, experiment)['precip']
                
                if run_on_gadi and np.isin(models_cmip6, dataset).any():
                    from cmip6_variables import *
                    precip = get_pr(institutes[model], model, experiment)['precip']
            else:
                precip = get_dsvariable('precip', dataset, experiment)
            

            # Calculate diagnostics and put into dataset
            quantile_threshold = 0.97
            conv_threshold = precip.quantile(quantile_threshold,dim=('lat','lon'),keep_attrs=True).mean(dim='time',keep_attrs=True)
            n = 8
            rome = calc_rome(precip, conv_threshold)
            rome_n = calc_rome_n(n, precip, conv_threshold)
            ds_rome = xr.Dataset(
                data_vars = {'rome':rome, 
                             'rome_n':rome_n},
                attrs = {'description': 'ROME based on all and the {} largest objects in the scene for each day'.format(n)}                  
                )
            ds_numberIndex = calc_numberIndex(precip, conv_threshold)
            ds_oAreaAndPr = calc_oAreaAndPr(precip, conv_threshold)



            # save
            save_rome = False
            save_numberIndex = False
            save_oAreaAndPr = False
            
            if np.isin(models_cmip5, dataset).any():
                folder_save = '{}/data/cmip5/metrics_cmip5_{}'.format(resolutions[0])
            if np.isin(models_cmip6, dataset).any():
                folder_save = '{}/data/cmip6/metrics_cmip6_{}'.format(resolutions[0])
            if np.isin(observations, dataset).any():
                folder_save = '{}/data/obs/metrics_obs_{}'.format(resolutions[0])

            if save_rome:
                fileName = dataset + '_rome_' + experiment + '_' + resolutions[0] + '.nc'              
                save_file(ds_rome, folder_save, fileName)

            if save_numberIndex:
                fileName = dataset + '_numberIndex_' + experiment + '_' + resolutions[0] + '.nc'
                save_file(ds_numberIndex, folder_save, fileName) 

            if save_oAreaAndPr:
                fileName = dataset + '_oAreaAndPr_' + experiment + '_' + resolutions[0] + '.nc'
                save_file(ds_oAreaAndPr, folder_save, fileName)


        stop = timeit.default_timer()
        print('dataset: {} took {} minutes to finsih'.format(dataset, (stop-start)/60))











