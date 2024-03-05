''' 
# ------------------------
#      ROME calc
# ------------------------
ROME - Radar Organization MEtric 
Evaluated from unique object pair weight: 
q(A_a, A_b, A_d) = A_a + min(1, A_b / A_d) * A_b
where
object  - contigous convective regions
A_a     - larger area of pair
A_b     - smaller area of pair
A_d     - squared shortest distance between pair boundaries (km)
ROME = mean(q)

to use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-calc')
import conv_org.rome_calc as rC
aList = calc_rome(conv_obj, obj_id, dim)
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ---------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
# import os
# import sys
# sys.path.insert(0, f'{os.getcwd()}/util-data')



# --------------------
#  General operations
# --------------------
# ------------------------------------------------------------------------------------ Get dimensions ----------------------------------------------------------------------------------------------------- #
class dims_class():
    R = 6371        # radius of earth
    g = 9.81        # gravitaional constant
    c_p = 1.005     # specific heat capacity
    L_v = 2.256e6   # latent heat of vaporization
    def __init__(self, da):
        self.lat, self.lon       = da['lat'].data, da['lon'].data
        self.lonm, self.latm     = np.meshgrid(self.lon, self.lat)
        self.dlat, self.dlon     = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
        self.aream               = np.cos(np.deg2rad(self.latm))*np.float64(self.dlon*self.dlat*self.R**2*(np.pi/180)**2) # area of domain
        self.latm3d, self.lonm3d = np.expand_dims(self.latm,axis=2), np.expand_dims(self.lonm,axis=2)                     # used for broadcasting
        self.aream3d             = np.expand_dims(self.aream,axis=2)


# ---------------------------------------------------------------------------------- Great circle distance ----------------------------------------------------------------------------------------------------- #
def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula)
    input: 
    lon range: [-180, 180]
    lat range: [-90, 90]
    (Takes vectorized input) 

    Formula:
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 
    where 
    phi -latitutde
    lambda - longitude
    '''
    R = 6371                    # radius of earth in km
    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180) # function requires 
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2 # Haversine formula
    h = np.clip(h, 0, 1)
    result =  2 * R * np.arcsin(np.sqrt(h))
    return result



# ----------
#   ROME
# ----------
# ---------------------------------------------------------------------------------------- one scene ----------------------------------------------------------------------------------------------------- #
def calc_rome_scene(conv_obj, labels, dim):
    ''' ROME (RAdar Organization MEtric) '''
    latm3d, lonm3d = dim.latm3d, dim.lonm3d                                                                 # latm3d, lonm3d will be extended in the third dimension with the height equal to the number of gridboxes in an object
    shape_L = np.shape(conv_obj)
    if len(labels) ==1:
        rome_allPairs = np.sum((conv_obj==labels)*1 * dim.aream)                                            # ROME = area of object if singular object
    else:
        rome_allPairs = []                                                                                  # combinations without repetition n! / (k!(n-k)!) (quicker with adding to empty list though)
        for idx, labeli in enumerate(labels[0:-1]):                                                         # compare object i and object j (unique pairs)
            I, J = zip(*np.argwhere(conv_obj==labeli))                                                      # find coordinates of object i
            I, J = list(I), list(J)
            oi_area = np.sum(np.squeeze(dim.aream)[I,J])                                                    # find area of object i
            Ni = len(I)                                                                                     # find number of gridboxes in object i
            lati3d = np.tile(dim.lat[I], reps =[shape_L[0], shape_L[1], 1])                                 # replicate each gridbox lon and lat into two 3D matrices: Ni 2D slices in the shape of L
            loni3d = np.tile(dim.lon[J], reps =[shape_L[0], shape_L[1], 1])
            if Ni > np.shape(lonm3d)[2]:                                                                    # create corresponding 3D matrix from Ni copies of latm, lonm (this metrix only needs to be recreated when Ni increases from previous loop)
                lonm3d = np.tile(lonm3d[:,:,0:1],reps =[1, 1, Ni])
                latm3d = np.tile(latm3d[:,:,0:1],reps =[1, 1, Ni])
            distancei3d = haversine_dist(lati3d, loni3d, latm3d[:,:,0:Ni], lonm3d[:,:,0:Ni])                # distance from gridbox in object i to every other point in the domain
            distancem = np.amin(distancei3d, axis=2)                                                        # minimum in the third dimension gives shortest distance from object i to every other point in the domain
            for labelj in labels[idx+1:]:                                                                   # the shortest distance from object i to object j will be the minimum of the coordinates of object j in distancem
                I, J = zip(*np.argwhere(conv_obj==labelj))                                                  # find coordinates of object j
                oj_area = np.sum(dim.aream[I,J])                                                            # find area of object j
                large_area = np.maximum(oi_area, oj_area) 
                small_area = np.minimum(oi_area, oj_area)
                rome_pair = large_area + np.minimum(small_area, (small_area/np.amin(distancem[I,J]))**2)    # ROME of unique pair
                rome_allPairs.append(rome_pair)
    return np.mean(np.array(rome_allPairs))


# ------------------------------------------------------------------------------------- organize result ----------------------------------------------------------------------------------------------------- #
def calc_rome(conv_obj, obj_id, dim):
    print('rome_calc started')
    rome = []
    for timestep in np.arange(0,2): #len(conv_obj.time.data)
        print(f'\t Processing {str(conv_obj.time.data[timestep])[:-8]} ...')
        scene = conv_obj.isel(time = timestep).data                                 # select scene and load data
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').values        # index of objects in scene (can be masked to select subset)
        rome.append(calc_rome_scene(scene, labels_scene, dim))
    return xr.DataArray(rome, dims = ['time'], coords = {'time': conv_obj.isel(time=slice(0,2)).time.data})



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('rome test starting')
    import os
    import sys
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data             as mD
    import variable_calc            as vC
    import var_calc.conv_obj_var    as cO
    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.line_plot as lP

    def pad_length_difference(da, max_length = 2): # max_length = 11000
        ''' When creating time series metrics, differnet dataset will have different lenghts.
            To put metrics from all datasets into one xarray Dataset the metric is padded with nan '''
        current_length = da.sizes['time']
        da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
        if current_length < max_length:
            padding = xr.DataArray(np.full((max_length - current_length,), np.nan), dims=['time'], coords={'time': np.arange(current_length, max_length)})
            da = xr.concat([da, padding], dim='time')
        return da

    switch = {'fixed_area': False}

    switch_test = {
        'delete_previous_plots':    True,
        'great_cricle_distance':    False,
        'rome':                     False,
        'rome_subset':              False,
        }
    
    experiment = cD.experiments[0]

    ds_rome = xr.Dataset()
    ds_rome_subset = xr.Dataset()
    for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
        # -------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
        conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = {'test_sample': True, 'ocean_mask': False}, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
        obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = {'test_sample': True, 'ocean_mask': False}, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
        conv = xr.where(conv_obj>0, 1, 0)
        dim = dims_class(conv_obj)
        conv.load()
        conv_obj.load()
        obj_id.load()
        metric_id_mask = cO.create_mock_metric_mask(obj_id)
        conv_obj_subset = cO.get_obj_subset(conv_obj, obj_id, metric_id_mask)

        # print(f'conv: \n {conv}')
        # print(f'conv_obj: \n {conv_obj}')
        # print(f'obj_id: \n {obj_id}')
        # print(f'metric id mask: \n {metric_id_mask}')
        # print(f'metric id mask: \n {conv_obj_subset}')
        # print(f'unique subset: \n {np.unique(conv_obj_subset)}')
        # exit()


        # --------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
        print(f'dataset: {dataset}')
        if switch_test['great_cricle_distance']:
            lat1 = 30
            lon1 = 30
            lat2 = 15
            lon2 = 15
            dist = haversine_dist(lat1, lon1, lat2, lon2)
            print(dist)

        if switch_test['rome']:
            # print(obj_id)
            rome = calc_rome(conv_obj, obj_id, dim)
            ds_rome[dataset] = pad_length_difference(rome)

        if switch_test['rome_subset']:
            obj_id_masked = obj_id * metric_id_mask
            rome = calc_rome(conv_obj, obj_id_masked, dim)
            ds_rome_subset[dataset] = pad_length_difference(rome)


        # ------------------------------------------------------------------------------------------- Plot --------------------------------------------------------------------------------------------------- #
        if switch_test['rome']:
            print(ds_rome)

        if switch_test['rome_subset']:
            print(ds_rome_subset)


















