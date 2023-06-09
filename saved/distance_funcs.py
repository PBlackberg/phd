import numpy as np
import xarray as xr
import skimage.measure as skm # used for giving connecting gridboxes the same number
import pandas as pd


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
        labels = np.unique(L)[1:] # first unique value (0) is background
        rome = np.append(rome, rome_scene(L, labels, lat, lon, aream, latm3d, lonm3d))
    
    rome = xr.DataArray(
        data = rome,
        dims = ['time'],
        coords = {'time': precip.time.data},
        attrs = {'units':'km\u00b2'}
        )
    return rome





if __name__ == '__main__':
    orgScenes = np.zeros(shape = (4, 22, 128))
    time_range = pd.date_range("1970/01/01","1970/01/05",freq='D',inclusive='left')
    lat = np.linspace(-30, 30, 22)
    lon = np.linspace(0, 360, 128)

    orgScenes = xr.DataArray(
        data = orgScenes,
        dims=['time','lat', 'lon'],
        coords={'time': time_range, 'lat': lat, 'lon': lon}
        )

    # one object
    orgScenes[0, :, :] = 0
    orgScenes[0, 10:15, 60:70] = 2

    # two objects (across boundary)
    orgScenes[1, :, :] = 0
    orgScenes[1, 10:15, 60:70] = 2

    orgScenes[1, 5:8, 0:10] = 2
    orgScenes[1, 5:10, 125:] = 2

    # two objects (do not cross boundary, but distance between objects across boundary is closer)
    orgScenes[2, :, :] = 0
    orgScenes[2, 5:8, 2:10] = 2

    orgScenes[2, 10:15, 120:-5] = 2

    # multiple objects (including crossing boundary multiple times) (9 objects)
    orgScenes[3, :, :] = 0
    orgScenes[3, 10:15, 60:70] = 2

    orgScenes[3, 5:8, 0:10] = 2
    orgScenes[3, 5:10, 125:] = 2

    orgScenes[3, 17:20, 0:3] = 2
    orgScenes[3, 17:19, 125:] = 2

    orgScenes[3, 16:18, 15:19] = 2

    orgScenes[3, 3:5, 30:40] = 2

    orgScenes[3, 10:17, 92:95] = 2

    orgScenes[3, 6:7, 105:106] = 2

    orgScenes[3, 2:4, 80:85] = 2

    orgScenes[3, 18:20, 35:39] = 2


    conv_threshold = 0.97 # percentile precipitation rate as threshold for convective region

    rome = calc_rome(orgScenes, conv_threshold)
    print(rome)


























































