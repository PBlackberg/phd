''' 
# ------------------------
#    Distance matrix
# ------------------------
This script creates a matrix with dimensions (gridboxes, lat, lon)
Where len(gridboxes) = len(lat) * len(lon)
The matrix describes the closest distance from gridbox[i] to any other point in the domain
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_base as vB



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



# --------------------
#   Distance matrix
# --------------------
def get_distance_matrix(switch = {}, dataset = '', experiment = '', resolution = '', dim = ''):         
    ''' This matrix will have the size (len(lat) * len(lon), len(lat), len(lon))
    For coarse cmip6 model: (2816, 22, 128) '''                               
    if dim == '':
        pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily')
        dim = dims_class(pr)
        del pr
    I, J = zip(*np.argwhere(np.ones_like(dim.latm)))                    # All spatial coordinates (i, j)
    I, J = list(I), list(J)                                             # index structured as (lat1, lon1), (lat1, lon2) ..
    n = len(I)                                                          # number of gridboxes
    lati3d = np.tile(dim.lat[I], reps =[len(dim.lat), len(dim.lon), 1]) # size: (n, lat, lon) (all lat, lon are one latitude value)
    loni3d = np.tile(dim.lon[J], reps =[len(dim.lat), len(dim.lon), 1]) # size: (n, lat, lon) (all lat, lon are one longitude value)
    latm3d = np.tile(dim.latm3d[:,:,0:1],reps =[1, 1, n])               #       (n, lat, lon) (all lat, lon are all latitude values)
    lonm3d = np.tile(dim.lonm3d[:,:,0:1],reps =[1, 1, n])               #       (n, lat, lon) (all lat, lon are all longitude values)
    distance_matrix = haversine_dist(lati3d, loni3d, latm3d, lonm3d)    # distance from gridbox i to every other point in the domain
    return xr.DataArray(distance_matrix, dims = ['lat', 'lon', 'gridbox'], coords = {'lat': dim.lat, 'lon': dim.lon, 'gridbox': np.arange(0,n)})

def obj_distance(conv_obj, obj_id, distance_matrix, index):
    # for timestep in np.arange(0,2): #len(conv_obj.time.data)
    timestep = 0
    scene = conv_obj.isel(time = timestep)
    if not index == '':
        labels = obj_id.isel(time = timestep).compute()[index]
    else:
        labels = obj_id.isel(time = timestep).compute()
        # print('executes')
        # print(labels)
        # exit()

    # print(labels)
    scene_subset = scene.isin(labels)
    scene_subset = xr.where(scene_subset > 0, 1, 0)
    scene_subset = scene_subset.compute().data
    I, J = zip(*np.argwhere(scene_subset)) 
    I, J = list(I), list(J)
    I, J = np.array(I), np.array(J)
    # print(f'I: \n {I}')
    # print(f'J: \n {J}')
    n_list = I * len(distance_matrix['lon']) + J
    # print(f'n_list: \n {n_list}')
    # exit()
    distance_from_obj = distance_matrix.sel(gridbox = n_list).min(dim = 'gridbox')
    return distance_from_obj



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    print('Minimum distance matrix test started')
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import variable_calc as vC
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD  
    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot as mP

    switch = {
        'test_sample': True,
        }

    switch_test = {
        'delete_previous_plots':                True,
        'great_cricle_distance':                False,
        'distance_from_gridbox':                False,
        'plot_pr':                              True,
        'plot_objects':                         True,
        'plot_objects_subset':                  True,
        'distance_from_object':                 True,
        'distance_from_object_together':        True,
        'distance_from_object_together_lim':    True,
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None

    dataset = cD.datasets[0]
    experiment = cD.experiments[0]
    resolution = cD.resolutions[0]

    # --------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
    conv_obj, _ = vC.get_variable(switch_var = {'conv_obj': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
    obj_id, _ = vC.get_variable(switch_var = {'obj_id': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
    dim = dims_class(conv_obj)
    # distance_matrix = get_distance_matrix(dim = dim)
    distance_matrix = get_distance_matrix(switch, dataset, experiment, resolution, dim = '')
    # print(distance_matrix)


    # --------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
    ds_dist = xr.Dataset()
    ds_pr = xr.Dataset()
    ds_obj = xr.Dataset()
    ds_obj_subset = xr.Dataset()
    ds_obj_dist = xr.Dataset()
    ds_obj_dist_together = xr.Dataset()
    ds_obj_dist_together_lim = xr.Dataset()

    if switch_test['great_cricle_distance']:
        lat1 = 30
        lon1 = 30
        lat2 = 15
        lon2 = 15
        dist = haversine_dist(lat1, lon1, lat2, lon2)
        print(dist)

    if switch_test['distance_from_gridbox']:
        for gridbox in np.arange(0,2500,100):
            ds_dist[f'distance_matrix_{gridbox}'] = distance_matrix.isel(gridbox = gridbox)

    if switch_test['plot_pr']:
        pr, _ = vC.get_variable(switch_var = {'pr': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0], from_folder = True, re_process = False)
        # print(pr)
        ds_pr[dataset] = pr.isel(time = 0)

    if switch_test['plot_objects']:
        ds_obj[dataset] = conv_obj.isel(time=0)>0

    if switch_test['plot_objects_subset']:
        timestep = 0        
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').values
        for index in np.arange(0, len(labels_scene)):
            scene = conv_obj.isel(time = timestep)
            scene_subset = scene.isin(obj_id.isel(time = timestep).compute()[index])
            scene_subset = xr.where(scene_subset > 0, 1, 0)
            ds_obj_subset[f'{dataset}_{index}'] = scene_subset

    if switch_test['distance_from_object']:
        labels_scene = obj_id.isel(time = timestep).dropna(dim='obj').values
        for index in np.arange(0, len(labels_scene)):
            ds_obj_dist[f'object_{int(index)}'] = obj_distance(conv_obj, obj_id, distance_matrix, index)

    if switch_test['distance_from_object_together']:
        ds_obj_dist_together['objects'] = obj_distance(conv_obj, obj_id, distance_matrix, index = '')

    if switch_test['distance_from_object_together_lim']:
        obj_dist_together = obj_distance(conv_obj, obj_id, distance_matrix, index = '')
        ds_obj_dist_together_lim['objects'] = obj_dist_together.where((obj_dist_together > 0) & (obj_dist_together < 2000))


    # ------------------------------------------------------------------------------ Plot --------------------------------------------------------------------------------------------------- #
    # print(ds_dist)
    if switch_test['distance_from_gridbox']:
        ds = ds_dist
        label = 'distance [km]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'distance_matrix.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_pr']:
        ds = ds_pr
        label = 'precipitation [mm/day]'
        vmin = 0
        vmax = 20
        cmap = 'Blues'
        filename = f'pr_scene_{dataset}.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_objects']:
        ds = ds_obj
        label = 'convection [0,1]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'conv.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_objects_subset']:
        ds = ds_obj_subset
        label = 'convection [0,1]'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'obj_conv.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['distance_from_object']:
        ds = ds_obj_dist
        label = 'distance [km]'
        vmin = None
        vmax = None
        cmap = 'Blues_r'
        filename = f'obj_distance.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['distance_from_object_together']:
        ds = ds_obj_dist_together
        label = 'distance [km]'
        vmin = None
        vmax = None
        cmap = 'Blues_r'
        filename = f'obj_distance_together.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['distance_from_object_together_lim']:
        ds = ds_obj_dist_together_lim
        label = 'distance [km]'
        vmin = None
        vmax = None
        cmap = 'Blues_r'
        filename = f'obj_distance_together_lim.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

















