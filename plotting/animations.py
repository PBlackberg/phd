import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
from matplotlib import animation

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD # imports functions to get data from gadi

sys.path.insert(0, f'{folder_code}/plotting')
import map_plot_scene as mp # imports functions to plot maps


# -------------------------------------------------------------------------------------- Create scene ----------------------------------------------------------------------------------------------------- #

def find_timesteps(array):
    threshold_low = 0.1
    array_prctile = np.percentile(array, threshold_low)
    chosen_timesteps_low= np.squeeze(np.argwhere(array.data<=array_prctile))

    threshold_mid1 = 49.95
    threshold_mid2 = 50.05
    array_prctile_mid1 = np.percentile(array, threshold_mid1)
    array_prctile_mid2 = np.percentile(array, threshold_mid2)
    chosen_timesteps_mid = np.squeeze(np.argwhere((array.data >= array_prctile_mid1) & (array.data <= array_prctile_mid2)))

    threshold_high = 99.9
    rome_prctile = np.percentile(array, threshold_high)
    chosen_timesteps_high= np.squeeze(np.argwhere(array.data>=rome_prctile))

    chosen_timesteps = np.concatenate((chosen_timesteps_low, chosen_timesteps_mid, chosen_timesteps_high))
    return chosen_timesteps

def plot_ax_scene(frame, fig, da, chosen_timesteps, cmap, cbar_label, conv_threshold, dataset):
    lat = da.lat
    lon = da.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))

    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    pr_day = da.isel(time=chosen_timesteps[frame])
    pcm= ax.pcolormesh(lonm,latm, pr_day.where(pr_day>conv_threshold),transform=ccrs.PlateCarree(),zorder=0, cmap='Blues', vmin=0, vmax=80)

    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    mp.format_ticks(ax, labelsize = 11)
    mF.plot_axtitle(fig, ax, f'{dataset}  gradually increasing DOC', xpad = 0.005, ypad = 0.035, fontsize=15)
    mp.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = cbar_label, text_pad = 0.125)
    plt.close()
    return fig, ax


# ----------------------------------------------------------------------------------------- Animate ----------------------------------------------------------------------------------------------------- #

def animate(da, array, cmap, cbar_label, conv_threshold, dataset):    
    chosen_timesteps = find_timesteps(array)

    fig= plt.figure(figsize=(12, 4))
    ani = animation.FuncAnimation(
        fig,                          
        plot_ax_scene,                                           # name of the function
        frames = len(chosen_timesteps),                          # Could also be iterable or list
        interval = 500,                                          # ms between frames
        fargs=(fig, da, chosen_timesteps, cmap, cbar_label, conv_threshold, dataset)  # Pass the additional parameters here
        )
    return ani


# ----------------------------------------------------------------------------------- load data and run script ----------------------------------------------------------------------------------------------------- #

def load_metric_array(switch, dataset, timescale, resolution, folder_load):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    variable_type, metric, metric_option, _, _ = mV.find_general_metric_and_specify_cbar(switch)
    if timescale == 'monthly' and (metric == 'percentiles_pr' or metric_option == 'rome'):
        array = mV.load_metric(folder_load, 'org', 'rome', source, dataset, 'daily', experiment = mV.experiments[0], resolution=resolution)['rome']
        array = mF.resample_timeMean(array, timescale)
    else:
        array = mV.load_metric(folder_load, 'org', 'rome', source, dataset, timescale, experiment = mV.experiments[0], resolution=resolution)['rome']
    return array

def load_data(switch, source, dataset, experiment, timescale, resolution, folder_save):
    if switch['constructed_fields']:
        return cF.var2D
    elif switch['sample_data']:
        return mV.load_sample_data(f'{folder_save}/pr', source, dataset, 'pr', timescale, experiment, resolution)['pr']
    else:
        return gD.get_pr(source, dataset, experiment, timescale, resolution)
    
def run_animation(switch, datasets, timescale, resolution, folder_save = mV.folder_save):
    print(f'Plotting map_plot with {resolution} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    
    dataset = datasets[0]
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    da = load_data(switch, source, dataset, experiment=mV.experiments[0], timescale=timescale, resolution=resolution, folder_save=folder_save)
    array = load_metric_array(switch, dataset, timescale, resolution, folder_save)
    variable_type, metric, _, cmap, cbar_label = mV.find_general_metric_and_specify_cbar(switch)

    conv_percentile = 0.97
    conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')

    ani = animate(da, array, cmap, cbar_label, conv_threshold, dataset)

    fileName = f'{dataset}conv_doc_transition.mp4'
    path = f'{folder_save}/{fileName}'
    # path = f'{home}/Desktop/{fileName}'
    ani.save(path)


if __name__ == '__main__':

    start = timeit.default_timer()

    # choose which metrics to plot
    switch = {
        'constructed_fields':  False, 
        'sample_data':         True,

        'pr':                  True,
        'percentiles_pr':      False,
        'hur':                 False,
        'rlut':                False,

        'save':                True,
        }
    

    # plot and save figure
    run_animation(switch, 
                 datasets =    mV.datasets, 
                 timescale =   mV.timescales[0],
                 resolution =  mV.resolutions[0],
                 folder_save = f'{mV.folder_save[0]}'
                 )

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
















































































