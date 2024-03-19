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
import choose_datasets as cD                                                      



# --------------------
#      Get data
# --------------------
# -------------------------------------------------------------------------------- Find convective regions ----------------------------------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- Visualize convective regions and objects ----------------------------------------------------------------------------------------------------- #
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






# ------------------------------------------------------------------------------------------ Mean area ----------------------------------------------------------------------------------------------------- #
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


# -------------------------------------------------------------------------------------------- Object area ----------------------------------------------------------------------------------------------------- #
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
    
