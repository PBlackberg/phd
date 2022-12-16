import numpy as np
import xarray as xr
import skimage.measure as skm

from vars.myFuncs import *
from vars.pr_vars import *


def calc_rome(precip, conv_threshold):
    rome = []

    lat = precip.lat.data
    lon = precip.lon.data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)
    latm3d = np.expand_dims(latm,axis=2) # used for broadcasting
    lonm3d = np.expand_dims(lonm,axis=2)


    for day in np.arange(0,len(precip.time.data)):
        pr_day = precip.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        
        connect_boundary(L)
        labels = np.unique(L)[1:]

        rome = np.append(rome, rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d))
    
    rome = xr.DataArray(
        data = rome,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'km^2'}
        )

    return rome


def rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d):
    rome_allPairs = []

    shape_L = np.shape(L)
    
    if len(labels) ==1:
        rome_allPairs = np.sum((L==labels)*1 * aream)

    else:
        for idx, labeli in enumerate(labels[0:-1]):
            
            # find coordinates of object i
            I, J = zip(*np.argwhere(L==labeli))
            I = list(I)
            J = list(J)

            # area of object i
            oi_area = np.sum(np.squeeze(aream)[I,J])

            # shortest distance from object i, start by counting the number of gridboxes
            Ni = len(I)

            # replicate each gridbox lon and lat into Ni 2D slices with the shape of L
            lati3d = np.tile(lat[I],reps =[shape_L[0], shape_L[1], 1])
            loni3d = np.tile(lon[J],reps =[shape_L[0], shape_L[1], 1])


            # create corresponding 3D matrix from Ni copies of latm, lonm (this metrix only needs to be recreated when Ni increases from previous loop)
            if Ni > np.shape(lonm3d)[2]:
                lonm3d = np.tile(lonm3d[:,:,0:1],reps =[1, 1, Ni])
                latm3d = np.tile(latm3d[:,:,0:1],reps =[1, 1, Ni])


            # distance from gridboxes of object i to every other point in the domain
            distancei3d = haversine_dist(lati3d,loni3d,latm3d[:,:,0:Ni],lonm3d[:,:,0:Ni])

            # minimum in the third dimension gives shortest distance from object i to every other point in the domain
            distancem = np.amin(distancei3d, axis=2)
    
    
            # the shortest distance from object i to object j will be the minimum of the coordinates of object j in distancem
            for labelj in labels[idx+1:]:

                # coordinates of object j
                I, J = zip(*np.argwhere(L==labelj))

                # area of object j
                oj_area = np.sum(aream[I,J])

                # ROME of unique pair
                large_area = np.maximum(oi_area, oj_area)
                small_area = np.minimum(oi_area, oj_area)
                rome_pair = large_area + np.minimum(small_area, (small_area/np.amin(distancem[I,J]))**2)
                rome_allPairs = np.append(rome_allPairs, rome_pair)
                
    return np.mean(rome_allPairs)






def calc_rome_n(n, precip, conv_threshold): 
    rome_n = []

    lat = precip.lat.data
    lon = precip.lon.data
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
                 'units': 'km^2'}
        )

    return rome_n


def rome_nScene(n, o_areaScene, L, labels, lat, lon, aream, latm3d, lonm3d):

    if len(o_areaScene) <= n:
        labels_n = labels
    else:
        labels_n = labels[o_areaScene.argsort()[-n:]]

    return rome_scene(L, labels_n, lat, lon, aream, latm3d, lonm3d)





def calc_numberIndex(precip, conv_threshold):
    o_number, areaf = [], []

    lat = precip.lat.data
    lon = precip.lon.data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)


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
        data=areaf,
        dims=['time'],
        coords={'time': precip.time.data}
        )


    numberIndex = xr.Dataset(
        data = {'o_number': o_number, 
                'areaf': areaf}
                ) 

    return numberIndex





def calc_o_area_and_o_pr(precip, conv_threshold):
    o_pr, o_area = [], []

    lat = precip.lat.data
    lon = precip.lon.data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)

    aream3d = np.expand_dims(aream,axis=2) # used for broadcasting

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
        attrs = {'units':'km^2'}
        )

    o_pr = xr.DataArray(
        data = o_pr,
        attrs = {'descrption': 'area weighted precipitation in contiguous convective region',
                'units':'mm/day'}
        )

    o_area_pr = xr.Dataset(
        data = {'o_area': o_area, 
                'o_pr': o_pr}
                ) 

    return o_area_pr







if __name__ == '__main__':

    models = [
            # 'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     
            # 'GISS-E2-H',    # 3
            # 'bcc-csm1-1',   # 4
            # 'CNRM-CM5',     # 5
            # 'CCSM4',        # 6 # cannot concatanate files for historical run
            # 'HadGEM2-AO',   # 7
            # 'BNU-ESM',      # 8
            # 'EC-EARTH',     # 9
            # 'FGOALS-g2',    # 10
            # 'MPI-ESM-MR',   # 11
            # 'CMCC-CM',      # 12
            # 'inmcm4',       # 13
            # 'NorESM1-M',    # 14
            # 'CanESM2',      # 15 # slicing with .sel does not work, 'contains no datetime objects'
            # 'MIROC5',       # 16
            # 'HadGEM2-CC',   # 17
            # 'MRI-CGCM3',    # 18
            # 'CESM1-BGC'     # 19
            ]
    
    experiments = [
                'historical',
                # 'rcp85'
                ]


    for model in models:
        for experiment in experiments:

            haveData = False
            if haveData:
                folder = '/g/data/k10/cb4968/data/cmip5/ds'
                fileName = model + '_precip_' + experiment + '.nc'
                path = folder + '/' + fileName
                precip = xr.open_dataset(path).precip
            else:
                precip = get_pr(model, experiment).precip

            conv_threshold = pr97 = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True).mean(dim='time',keep_attrs=True)
            n = 8



            rome = calc_rome(precip, conv_threshold)

            saveit = False            
            if saveit:                
                dataSet = rome
                myFuncs.save_file(dataSet, folder, fileName)

            



            rome_n = calc_rome_n(n, precip, conv_threshold)

            saveit = False
            if saveit:
                dataSet = rome_n
                myFuncs.save_file(dataSet, folder, fileName)





            numberIndex = calc_numberIndex(precip, conv_threshold)

            saveit = False
            if saveit:
                dataSet = numberIndex
                myFuncs.save_file(dataSet, folder, fileName)





            o_area_pr = calc_o_area_and_o_pr(precip, conv_threshold)

            saveit = False
            if saveit:
                dataSet = o_area_pr
                myFuncs.save_file(dataSet, folder, fileName)












