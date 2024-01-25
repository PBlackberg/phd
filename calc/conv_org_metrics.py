''' 
# ------------------------
#  Organization metrics
# ------------------------
In this script organization metrics are calculated from convective regions.
The convective regions are binary 2D fields.
The convective regions are determined by precipitation rates exceeding a percentile threshold (on average 5% of the domain covered is the default)
'''



# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import skimage.measure as skm


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                                 
import myFuncs as mF                                



# --------------------
#      Get data
# --------------------
# -------------------------------------------------------------------------------- Find convective regions ----------------------------------------------------------------------------------------------------- #
def get_conv_threshold(da):
    ''' Gridboxes exceeding fixed precipitation rate threshold considered convective (average 5% of domain is default) '''
    conv_threshold = da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)
    conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
    return conv_threshold

def get_fixed_area_conv_threshold(da):
    ''' Gridboxes exceeding variable precipitation rate considered convective. Area is fixed to 5% each day '''
    return da.quantile(int(mV.conv_percentiles[0])*0.01, dim=('lat', 'lon'), keep_attrs=True)

@mF.timing_decorator()
def find_convective_regions(switch, dataset, experiment):
    da = mF.load_variable({'pr': True}, switch, dataset, experiment)
    # print(da)
    # exit()
    conv_threshold = get_conv_threshold(da) if not switch['fixed_area'] else get_fixed_area_conv_threshold(da)    
    conv_regions = (da > conv_threshold)*1
    return conv_regions


# --------------------------------------------------------------------------- Visualize convective regions and objects ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def get_conv_snapshot(da):
    ''' Convective region '''
    plot = False
    if plot:
        import myFuncs_plots as mFd     
        for timestep in np.arange(0, len(da.time.data)):
            fig = mFd.plot_scene(da.isel(time=timestep), ax_title = timestep)    #, vmin = 0, vmax = 60) #, cmap = 'RdBu')
            if mFd.show_plot(fig, show_type = 'cycle', cycle_time = 0.2):        # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)
                break
    return da.isel(time=0) 

@mF.timing_decorator()
def get_obj_snapshot(da):
    ''' Contihuous convective regions (including crossing lon boundary) '''
    plot = False
    if plot:
        import myFuncs_plots as mFd         
        for day in np.arange(0, len(da.time.data)):
            scene = skm.label(da.isel(time=day), background = 0, connectivity=2)
            fig = mFd.plot_scene(xr.DataArray(scene, dims=['lat', 'lon'], coords={'lat': da.lat.data, 'lon': da.lon.data}))
            if mFd.show_plot(fig, show_type = 'cycle', cycle_time = 0.2):                    # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)
                break
    scene = skm.label(da.isel(time=0), background = 0, connectivity=2)
    return xr.DataArray(scene, dims=['lat', 'lon'], coords={'lat': da.lat.data, 'lon': da.lon.data})


# ------------------------
#    Calculate metrics
# ------------------------
# ---------------------------------------------------------------------------------------- Areafraction ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def calc_areafraction(da, dim):
    ''' Areafraction convered by convection (with fixed precipitation rate, the convective area will fluctuate from day to day)'''
    areaf = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        areaf_scene = (np.sum(da.isel(time=day) * dim.aream)/np.sum(dim.aream))*100
        areaf[day] = areaf_scene
    return xr.DataArray(areaf, dims = ['time'], coords = {'time': da.time.data})


# -------------------------------------------------------------------------------------- NI (Number Index) ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def calc_ni(da):
    ''' Number of object in scene (a rough indicator of clumpiness) '''
    ni = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        da_day = skm.label(da.isel(time=day), background=0,connectivity=2)
        da_day = mF.connect_boundary(da_day)
        labels = np.unique(da_day)[1:]
        o_numberScene = len(labels)
        ni[day] = o_numberScene
    return xr.DataArray(ni, dims = ['time'], coords = {'time': da.time.data})


# ------------------------------------------------------------------------------------------ Mean area ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def calc_o_area_mean(da, dim):
    ''' Mean area of each contiguous convective region (object) in a day (similar to ROME (Radar Organization MEtric) when area is fixed) '''
    mean_area = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        da_day = skm.label(da.isel(time=day), background=0, connectivity=2)
        da_day = mF.connect_boundary(da_day)
        labels = np.unique(da_day)[1:]
        obj3d = np.stack([(da_day == label) for label in labels],axis=2)*1 # 3d matrix with each object in a scene being a binary 2d slice
        o_areaScene = np.sum(obj3d * dim.aream3d, axis=(0,1))
        mean_area[day] = np.sum(o_areaScene)/len(labels)
    return xr.DataArray(mean_area, dims = ['time'], coords = {'time': da.time.data})


# -------------------------------------------------------------------------------- ROME (Radar Organization MEtric) ----------------------------------------------------------------------------------------------------- #
def rome_scene(da, labels, dim):
    ''' Calculates ROME for a scene: 
        Average of unique pair weight: A_a + min(1, A_b / A_d) * A_b
        where
        A_a - larger area of pair
        A_b - smaller area of pair
        A_d - shortest distance between pair boundaries (km) '''
    latm3d, lonm3d = dim.latm3d, dim.lonm3d                                                                     # these are updated for each loop (extended in thrid dimension)
    shape_L = np.shape(da)
    if len(labels) ==1:
        rome_allPairs = np.sum((da==labels)*1 * dim.aream)                                                      # ROME = area of object if singular object
    else:
        rome_allPairs = [None] * int(np.math.factorial(len(labels)) / (2 * np.math.factorial((len(labels)-2)))) # combinations without repetition n! / (k!(n-k)!)
        idx_o = 0
        for idx, labeli in enumerate(labels[0:-1]):                                                             # compare object i and object j
            I, J = zip(*np.argwhere(da==labeli))                                                                # find coordinates of object i
            I, J = list(I), list(J)
            oi_area = np.sum(np.squeeze(dim.aream)[I,J])                                                        # find area of object i
            Ni = len(I)                                                                                         # find number of gridboxes in object i
            lati3d = np.tile(dim.lat[I],reps =[shape_L[0], shape_L[1], 1])                                      # replicate each gridbox lon and lat into Ni 2D slices in the shape of L
            loni3d = np.tile(dim.lon[J],reps =[shape_L[0], shape_L[1], 1])
            if Ni > np.shape(lonm3d)[2]:                                                                        # create corresponding 3D matrix from Ni copies of latm, lonm (this metrix only needs to be recreated when Ni increases from previous loop)
                lonm3d = np.tile(lonm3d[:,:,0:1],reps =[1, 1, Ni])
                latm3d = np.tile(latm3d[:,:,0:1],reps =[1, 1, Ni])
            distancei3d = mF.haversine_dist(lati3d, loni3d, latm3d[:,:,0:Ni], lonm3d[:,:,0:Ni])                 # distance from gridbox in object i to every other point in the domain
            distancem = np.amin(distancei3d, axis=2)                                                            # minimum in the third dimension gives shortest distance from object i to every other point in the domain
            for labelj in labels[idx+1:]:                                                                       # the shortest distance from object i to object j will be the minimum of the coordinates of object j in distancem
                I, J = zip(*np.argwhere(da==labelj))                                                            # find coordinates of object j
                oj_area = np.sum(dim.aream[I,J])                                                                # find area of object j
                large_area = np.maximum(oi_area, oj_area) 
                small_area = np.minimum(oi_area, oj_area)
                rome_pair = large_area + np.minimum(small_area, (small_area/np.amin(distancem[I,J]))**2)        # ROME of unique pair
                rome_allPairs[idx_o] = rome_pair
                idx_o += 1
    return np.mean(np.array(rome_allPairs))

@mF.timing_decorator()
def calc_rome(da, dim):
    ''' ROME (RAdar Organization MEtric) '''
    rome = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        print(f'\t Processing ... currently at: {da.time.data[day]}') if day % 365 == 0 else None
        da_day = skm.label(da.isel(time=day), background=0, connectivity=2)
        da_day = mF.connect_boundary(da_day)
        labels = np.unique(da_day)[1:]  # first unique value (zero) is background
        rome[day] = rome_scene(da_day, labels, dim)
    return xr.DataArray(rome, dims = ['time'], coords = {'time': da.time.data})

@mF.timing_decorator()
def calc_rome_n(da, dim, n=8): 
    ''' ROME_n
        Finds n largest objects and calls rome_scene of subset of objects '''
    rome_n = [None] * len(da.time.data)
    for day in np.arange(0,len(da.time.data)):
        print(f'\t Processing ... currently at: {da.time.data[day]}') if day % 365 == 0 else None
        da_day = skm.label(da.isel(time = day), background = 0, connectivity = 2)
        da_day = mF.connect_boundary(da_day)
        labels = np.unique(da_day)[1:]
        obj3d = np.stack([(da_day == label) for label in labels],axis=2)*1
        o_areaScene = np.sum(obj3d * dim.aream3d, axis=(0,1))
        labels_n = labels if len(labels) <= n else labels[o_areaScene.argsort()[-n:]]
        rome_n[day] = rome_scene(da_day, labels_n, dim)
    return xr.DataArray(rome_n, dims = ['time'], coords = {'time': da.time.data})


# -------------------------------------------------------------------------------------------- Object area ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def calc_o_area(da, dim):
    ''' Area of each contiguous convective regions (objects). Used to create probability distribution of sizes of convective blobs '''
    # o_area = [None] * len(da.time.data) * len(dim.lat) * len(dim.lon)
    # idx = 0
    o_area = []
    for day in np.arange(0,len(da.time.data)):
        da_day = skm.label(da.isel(time=day), background=0,connectivity=2)
        da_day = mF.connect_boundary(da_day)
        labels = np.unique(da_day)[1:]
        obj3d = np.stack([(da_day == label) for label in labels],axis=2)*1
        o_areaScene = np.sum(obj3d * dim.aream3d, axis=(0,1))
        o_area.extend(o_areaScene)
    #     o_area[idx:idx+len(labels)] = o_areaScene
    #     idx += len(labels)  # python is non-inclusive to the right, so doesn't overwrite (.extend() method is quicker)
    # o_area = o_area[:idx]
    return xr.DataArray(o_area, dims = ['obj'], coords = {'obj': np.arange(0, len(o_area))})
    

# ------------------------------------------------------------------------------------------- Object heatmap ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def calc_o_heatmap(da):
    ''' Frequency of occurence of objects in individual gridboxes in tropical scene '''
    return da.sum(dim= 'time') / len(da.time.data)


# ------------------------------------------------------------------------------------------------ I_org ----------------------------------------------------------------------------------------------------- #
# def i_org():
#     ''



# ------------------------
#   Run / save metrics
# ------------------------
# ------------------------------------------------------------------------------------- Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def calc_metric(switch_metric, da):
    ''' Calls organization metric from convective regions '''
    dim = mF.dims_class(da)
    for metric_name in [k for k, v in switch_metric.items() if v]:
        metric = None
        metric = get_conv_snapshot(da)          if metric_name == 'conv_snapshot'   else metric
        metric = get_obj_snapshot(da)           if metric_name == 'obj_snapshot'    else metric
        metric = calc_areafraction(da, dim)     if metric_name == 'areafraction'    else metric
        metric = calc_ni(da)                    if metric_name == 'ni'              else metric
        metric = calc_o_area_mean(da, dim)      if metric_name == 'o_area_mean'     else metric
        metric = calc_rome(da, dim)             if metric_name == 'rome'            else metric
        metric = calc_rome_n(da, dim, n = 8)    if metric_name == 'rome_n'          else metric
        metric = calc_o_area(da, dim)           if metric_name == 'o_area'          else metric
        metric = calc_o_heatmap(da)             if metric_name == 'o_heatmap'       else metric

        metric_name = f'{metric_name}_{mV.conv_percentiles[0]}thprctile' if not switch['fixed_area'] else f'{metric_name}_{mV.conv_percentiles[0]}thprctile_fixed_area'
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def run_org_metrics(switch_metric, switch):
    print(f'variable: Binary matrix of gridboxes exceeding {mV.conv_percentiles[0]}th percentile precipitation threshold, using {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'metric: {[key for key, value in switch_metric.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for dataset, experiment in mF.run_dataset():
        da = find_convective_regions(switch, dataset, experiment)
        for metric, metric_name in calc_metric(switch_metric, da):
            mF.save_metric(switch, 'org', dataset, experiment, metric, metric_name)
            print(f'\t\t\t{metric_name} saved') if switch['save_folder_desktop'] or switch['save_scratch'] or switch['save'] else None


# ---------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_metric = {                                                                   # metric
        'conv_snapshot':    False,   'obj_snapshot':     False,  'areafraction': False, # conceptual visualization of data
        'ni':               False,                                                      # Number index
        'o_area_mean':      False,                                                      # object mean area                     
        'rome':             False,  'rome_n':           False,                          # ROME
        'o_area':           False,                                                      # object area 
        'o_heatmap':        False,                                                      # object location
        }

    switch = {                                                                          # settings
        'fixed_area':           False,                                                  # conv_threshold type (fixed area instead of fixed precipitation rate threshold)
        'constructed_fields':   False, 'test_sample':   False,                          # For testing
        'save_folder_desktop':  False, 'save_scratch':  False, 'save': False            # Save
        }
    
    run_org_metrics(switch_metric, switch)
    
