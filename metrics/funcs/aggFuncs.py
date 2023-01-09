import numpy as np
import xarray as xr
import skimage.measure as skm




# Connect objects across boundary (objects that touch across lon=0, lon=360 boundary are the same object) (takes array(lat, lon))
def connect_boundary(array):
    s = np.shape(array)
    for row in np.arange(0,s[0]):
        if array[row,0]>0 and array[row,-1]>0:
            array[array==array[row,0]] = min(array[row,0],array[row,-1])
            array[array==array[row,-1]] = min(array[row,0],array[row,-1])



# Haversine formula (Great circle distance) (takes vectorized input)
def haversine_dist(lat1, lon1, lat2, lon2):

   # radius of earth in km
    R = 6371

    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180)     
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)

    # Haversine formula
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2

    # distance from Haversine function:
    # (1) h = sin(theta/2)^2

    # central angle, theta:
    # (2) theta = d_{great circle} / R
    
    # (1) in (2) and rearrange for d gives
    # d = R * sin^-1(sqrt(h))*2 

    return 2 * R * np.arcsin(np.sqrt(h))



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
        data_vars = {'o_number': o_number, 
                     'areaf': areaf}
        ) 

    return numberIndex





def calc_rEff(area): # input as m^2
    return np.sqrt(area/np.pi)*1e-3

def calc_pwad_bin(idx, o_area, o_pr): #, obj_area, obj_pr
    return np.sum(idx*o_area*o_pr)/(np.sum(o_area*o_pr))


def calc_pwad(precip, conv_threshold):
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
        

    o_r = calc_rEff(o_area)

    # define the bin width of the mean gridbox area (as effective radius)
    bin_width = calc_rEff(np.mean(aream))
    bin_end = calc_rEff(np.max(o_area))
    bins = np.arange(0, bin_end+bin_width, bin_width)
    bins_mid = np.append(0,bins*0.5*bin_width) # place the datapoints in the middle of a bin


    # place fractional amount of precipitation fallling in respective bin
    pwad_bins = []
    for i in np.arange(0,len(bins)-1):
        idx = (o_r>bins[i]) & (o_r<=bins[i+1])
        pwad_bins = np.append(pwad_bins, calc_pwad_bin(idx, o_area, o_pr))

    # close the distribution by describing zero objects in bins smaller or greater than min, max
    pwad_bins = np.append(0, pwad_bins)
    pwad_bins = np.append(pwad_bins,0)



    o_area = xr.DataArray(
        data = o_area,
        attrs = {'units':'km^2'}
        )

    o_pr = xr.DataArray(
        data = o_pr,
        attrs = {'descrption': 'area weighted precipitation in contiguous convective region',
                'units':'mm/day'}
        )

    pwad_bins = xr.DataArray(
        data = pwad_bins,
        attrs = {'descrption': 'fractional distribution of precipitation into bins of area in terms of effective radius',
                'units':''}
        )

    bins_mid = xr.DataArray(
        data = bins_mid,
        attrs = {'descrption': 'datapoints for middle of effective radius bins',
                'units':'km'}
        )

    pwad = xr.Dataset(
        data_vars = {'pwad': pwad_bins, 
                     'bins_mid': bins_mid}
        )
                    #  'o_area': o_area,
                    #  'o_pr': o_pr}
        
    return pwad







if __name__ == '__main__':

    from os.path import expanduser
    home = expanduser("~")
    from vars.myFuncs import *



    models = [
        # 'IPSL-CM5A-MR', # 1
        'GFDL-CM3',     # 2
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


    switch = {
        'local_files': True, 
        'nci_files': False, 
    }


    for model in models:
        for experiment in experiments:

            if switch['local_files']:
                folder = home + '/Documents/data/cmip5/ds'
                fileName = model + '_precip_' + experiment + '.nc'
                path = folder + '/' + fileName
                ds = xr.open_dataset(path)
                precip = ds.precip*60*60*24
                precip.attrs['units']= 'mm/day'
                folder = home + '/Documents/data/cmip5/' + model

            if switch['nci_files']:
                from vars.prVars import *
                precip = get_pr(model, experiment).precip
                folder = '/g/data/k10/cb4968/data/cmip5/'+ model



            conv_threshold = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True).mean(dim='time',keep_attrs=True)
            n = 8



            rome = calc_rome(precip, conv_threshold)
            rome_n = calc_rome_n(n, precip, conv_threshold)

            numberIndex = calc_numberIndex(precip, conv_threshold)

            pwad = calc_pwad(precip, conv_threshold)



            saveit = False            
            if saveit:  
                fileName = model + '_rome_' + experiment + '.nc'              
                dataset = xr.Dataset(
                    data_vars = {'rome':rome, 
                                 'rome_n':rome_n},
                    attrs = {'description': 'ROME based on all and the {} largest contiguous convective regions in the scene for each day'.format(n),
                             'units':'km^2'}                  
                        )

                save_file(dataset, folder, fileName)


            saveit = False
            if saveit:
                fileName = model + '_numberIndex_' + experiment + '.nc'
                dataset = numberIndex

                save_file(dataset, folder, fileName) 


            saveit = False
            if saveit:
                fileName = model + '_pwad_' + experiment + '.nc'
                dataset = pwad

                save_file(dataset, folder, fileName) 






