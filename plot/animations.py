import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeat
import matplotlib.pyplot as plt
from matplotlib import animation

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import myFuncs as mF # imports common operators
import myVars as mV # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD # imports functions to get data from gadi


# -------------------------------------------------------------------------------------- Create scene ----------------------------------------------------------------------------------------------------- #

def find_timesteps(array):
    # threshold_low = 0.1
    # array_prctile = np.percentile(array, threshold_low)
    # chosen_timesteps_low= np.squeeze(np.argwhere(array.data<=array_prctile))

    # threshold_mid1 = 49.95
    # threshold_mid2 = 50.05
    # array_prctile_mid1 = np.percentile(array, threshold_mid1)
    # array_prctile_mid2 = np.percentile(array, threshold_mid2)
    # chosen_timesteps_mid = np.squeeze(np.argwhere((array.data >= array_prctile_mid1) & (array.data <= array_prctile_mid2)))

    # threshold_high = 99.9
    threshold_high = 99.5
    rome_prctile = np.percentile(array, threshold_high)
    chosen_timesteps_high= np.squeeze(np.argwhere(array.data>=rome_prctile))

    # chosen_timesteps = np.concatenate((chosen_timesteps_low, chosen_timesteps_mid, chosen_timesteps_high))

    chosen_timesteps = chosen_timesteps_high

    return chosen_timesteps

def plot_ax_scene(frame, fig, da, chosen_timesteps, cmap, cbar_label, conv_threshold, dataset):
    lat = da.lat
    lon = da.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))

    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    pr_day = da.isel(time=chosen_timesteps[frame])
    # threshold = pr_day.quantile(conv_threshold, dim=('lat', 'lon'), keep_attrs=True)
    pcm= ax.pcolormesh(lonm,latm, pr_day.where(pr_day>conv_threshold),transform=ccrs.PlateCarree(),zorder=0, cmap='Blues', vmin=0, vmax=80)

    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    mF.format_ticks(ax, labelsize = 11)
    # mF.plot_axtitle(fig, ax, f'{dataset}  gradually increasing DOC {mV.experiments[0]}', xpad = 0.005, ypad = 0.035, fontsize=15)
    mF.plot_axtitle(fig, ax, f'{dataset}  top 2.5% DOC {mV.experiments[0]}', xpad = 0.005, ypad = 0.035, fontsize=15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = cbar_label, text_pad = 0.125)
    plt.close()
    return fig, ax


# ----------------------------------------------------------------------------------------- Animate ----------------------------------------------------------------------------------------------------- #

def animate(da, threshold, array, dataset, metric):
    chosen_timesteps = find_timesteps(array)

    fig= plt.figure(figsize=(12, 4))
    ani = animation.FuncAnimation(
        fig,                          
        plot_ax_scene,                                           # name of the function
        frames = len(chosen_timesteps),                          # Could also be iterable or list
        interval = 500,                                          # ms between frames
        fargs=(fig, da, chosen_timesteps, metric.cmap, metric.label, threshold, dataset)  # Pass the additional parameters here
        )
    return ani

# ----------------------------------------------------------------------------------- load data and run script ----------------------------------------------------------------------------------------------------- #

def load_metric_array(source, dataset, metric):
    folder = metric.get_metric_folder(mV.folder_save[0], metric.name, source)
    # folder = f'{mV.folder_save[0]}/org/metrics/rome_equal_area/{source}'
    # filename = f'{dataset}_rome_equal_area_daily_{mV.experiments[0]}_{mV.resolutions[0]}.nc'

    folder = f'{mV.folder_save[0]}/org/metrics/rome/{source}'
    filename = f'{dataset}_rome_daily_{mV.experiments[0]}_{mV.resolutions[0]}.nc'
    array = xr.open_dataset(f'{folder}/{filename}')['rome']
    return array

def load_data(switch, source, dataset):
    da = cF.var2D if switch['constructed_fields'] else None
    if switch['sample_data']:
        folder = f'/Users/cbla0002/Documents/data/pr/sample_data/{source}'
        filename = f'{dataset}_pr_daily_{mV.experiments[0]}_{mV.resolutions[0]}.nc'
        da = xr.open_dataset(folder + '/' + filename)['pr']
    da = gD.get_pr(source, dataset, mV.timescales[0], mV.experiments[0], mV.resolutions[0]) if switch['gadi_data'] else da
    return da

def run_animation(switch):
    if not switch['run']:
        return
    print(f'Plotting map_plot with {mV.resolutions[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    
    dataset = mV.datasets[0]
    source = mF.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    metric = mF.get_metric_object(switch)
    da = load_data(switch, source, dataset)
    array = load_metric_array(source, dataset, metric)

    conv_percentile = 0.97
    conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True).mean(dim='time')
    # conv_threshold = 0.97

    ani = animate(da, conv_threshold, array, dataset, metric)

    fileName = f'{dataset}_conv_doc_transition_{mV.experiments[0]}.mp4'
    # path = f'{mV.folder_save[0]}/{fileName}'
    path = f'{home}/Desktop/{fileName}'
    ani.save(path)



if __name__ == '__main__':
    run_animation(switch = {
        'constructed_fields':  False, 
        'sample_data':         True,
        'gadi_data':           False,

        'pr':                  True,
        'percentiles_pr':      False,
        'hur':                 False,
        'rlut':                False,

        'run':                 True,
        'save':                True,
        }
    )















































































