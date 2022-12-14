import numpy as np
import xarray as xr
import skimage.measure as skm
import myFuncs



def calc_rome(precip, listOfdays, conv_threshold):
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
    aream3d = np.expand_dims(aream,axis=2) # (usef later for n largest)    


    for day in listOfdays:
        pr_day = precip.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        myFuncs.connect_boundary(L)
        
        labels = np.unique(L)[1:]

        rome = np.append(rome, rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d))
    
    rome = xr.DataArray(
        data=rome,
        dims=['time'],
        coords={'time': precip.time.data[0:len(listOfdays)]},
        attrs={'units':'km^2'}
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
            distancei3d = myFuncs.haversine_dist(lati3d,loni3d,latm3d[:,:,0:Ni],lonm3d[:,:,0:Ni])

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








def calc_rome_n(n, precip, listOfdays, conv_threshold): 
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

    for day in listOfdays:
        pr_day = precip.isel(time=day)
        
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        myFuncs.connect_boundary(L)
        labels = np.unique(L)[1:]

        obj3d = np.stack([(L==label) for label in labels],axis=2)*1
        o_areaScene = np.sum(obj3d * aream3d, axis=(0,1))

        rome_n = np.append(rome_n, rome_nScene(n, o_areaScene, L, labels, lat, lon, aream, latm3d, lonm3d))
    

    rome_n = xr.DataArray(
        data=rome_n,
        dims=['time'],
        coords={'time': precip.time.data[0:len(listOfdays)]},
        attrs={'description':'rome calculated from {} largest contigiuos convetive areas'.format(n),
        'units': 'km^2'}
        )

    return rome_n


def rome_nScene(n, o_areaScene, L, labels, lat, lon, aream, latm3d, lonm3d):

    if len(o_areaScene) <= n:
        labels_n = labels
    else:
        labels_n = labels[o_areaScene.argsort()[-n:]]

    return rome_scene(L, labels_n, lat, lon, aream, latm3d, lonm3d)







def calc_numberIndex(precip, listOfdays, conv_threshold):
    numberIndex, areaf = [], []

    lat = precip.lat.data
    lon = precip.lon.data
    lonm,latm = np.meshgrid(lon,lat)
    dlat = (lat[1]-lat[0])
    dlon = (lon[1]-lon[0])
    R = 6371
    aream = np.cos(np.deg2rad(latm))*np.float64(dlon*dlat*R**2*(np.pi/180)**2)


    for day in listOfdays:
        pr_day = precip.isel(time=day)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        myFuncs.connect_boundary(L)
        labels = np.unique(L)[1:]

        numberIndex_scene = len(labels)
        
        conv_day = (pr_day.where(pr_day>=conv_threshold,0)>0)*1
        areaf_scene = np.sum(conv_day * aream)/np.sum(aream)
        
        numberIndex = np.append(numberIndex, numberIndex_scene)
        areaf = np.append(areaf, areaf_scene)


    numberIndex = xr.DataArray(
        data=numberIndex,
        dims=['time'],
        coords={'time': precip.time.data[0:len(listOfdays)]},
        attrs={'units':'Nb'}
        )

    areaf = xr.DataArray(
        data=areaf,
        dims=['time'],
        coords={'time': precip.time.data[0:len(listOfdays)]}
        )

    return numberIndex, areaf







def calc_area_pr(precip, listOfdays, conv_threshold):
    o_pr, o_area = [], []

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

    for day in listOfdays:
        pr_day = precip.isel(time=day)
        pr_day3d = np.expand_dims(pr_day,axis=2)
        L = skm.label(pr_day.where(pr_day>=conv_threshold,0)>0, background=0,connectivity=2)
        myFuncs.connect_boundary(L)
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

    return o_area, o_pr
























