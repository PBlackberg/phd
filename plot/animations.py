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
sys.path.insert(0, f'{os.getcwd()}/util')
import myFuncs as mF            # imports common operators
import myVars as mV             # imports common variables
import constructed_fields as cF # imports fields for testing
import get_data as gD           # imports functions to get data from gadi


# -------------------------------------------------------------------------------------- animate / format plot ----------------------------------------------------------------------------------------------------- #

def plot_ax_scene(switch, fig, da_0, da_1, timesteps, frame):
    lat, lon = da_0.lat, da_0.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    timestep = timesteps[frame]

    ax.pcolormesh(lonm,latm, da_0.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap='Blues', vmin=0, vmax=80)
    pcm = ax.pcolormesh(lonm,latm, da_0.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap='Reds', vmin=0, vmax=80)

    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    mF.format_ticks(ax, labelsize = 11)
    mF.plot_axtitle(fig, ax, f'{mV.datasets[0]}  gradually increasing DOC {mV.experiments[0]}', xpad = 0.005, ypad = 0.035, fontsize=15)
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = cbar_label, text_pad = 0.125)
    plt.close()
    return fig, ax

def animate(da, threshold, array, dataset, metric):
    chosen_timesteps = get_timesteps(array)

    fig= plt.figure(figsize=(12, 4))
    ani = animation.FuncAnimation(
        fig,                          
        plot_ax_scene,                                                                    # name of the function
        frames = len(chosen_timesteps),                                                   # Could also be iterable or list
        interval = 500,                                                                   # ms between frames
        fargs=(fig, da, chosen_timesteps, metric.cmap, metric.label, threshold, dataset)  # Pass the additional parameters here
        )
    return ani

# ----------------------------------------------------------------------------------- load data and run script ----------------------------------------------------------------------------------------------------- #

def calc_conv_threshold(da, conv_percentile, fixed_area): # conv_threshold is number [0,1]
    ''' Convection can be based on fixed precipitation rate threshold or fixed areafraction (same time-mean area) '''
    conv_threshold = da.quantile(conv_percentile, dim=('lat', 'lon'), keep_attrs=True)
    if not fixed_area:
        conv_threshold = xr.DataArray(data = conv_threshold.mean(dim='time').data * np.ones(shape = len(da.time)), dims = 'time', coords = {'time': da.time.data}) 
    return conv_threshold

def load_data(switch, variable):
    source = mF.find_source(mV.datasets[0], mV.models_cmip5, mV.models_cmip6, mV.observations)
    da = cF.var2d                                                                                                       if switch['constructed_fields'] else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/{variable.variable_type}/sample_data/{source}/ \
        {mV.datasets[0]}_{variable.name}_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')[variable.name] if switch['sample_data'] else da
    da = gD.get_pr(source, mV.datasets[0], mV.timescales[0], mV.experiments[0], mV.resolutions[0])                      if switch['gadi_data'] else da
    return da

def get_da(switch, variable):
    if variable.name == 'obj':
        da = load_data(switch, variable)
        conv_threshold = calc_conv_threshold(da, conv_percentile = int(mV.conv_percentiles[0]) * 0.01, fixed_area = switch['fixed_area'])
        da = da.where(da >= conv_threshold)

    if variable.name == 'pr99':
        da = load_data(switch, variable)
        conv_threshold = calc_conv_threshold(da, conv_percentile = 0.99, fixed_area = True)
        da = da.where(da >= conv_threshold)

    if variable.name in ['pr', 'rlut', 'hur']:
        da = load_data(switch, variable)
    

def get_timesteps(switch, array):
    source = mF.find_source(mV.datasets[0], mV.models_cmip5, mV.models_cmip6, mV.observations)
    array = xr.open_dataset(f'{mV.folder_save[0]}/org/metrics/rome/{source}/{mV.datasets[0]}_rome_daily_{mV.experiments[0]}_{mV.resolutions[0]}.nc')['rome']
    if switch['transition']:
        threshold_low = 0.1
        array_prctile = np.percentile(array, threshold_low)
        chosen_timesteps_low= np.squeeze(np.argwhere(array.data<=array_prctile))

        threshold_mid1 = 49.95
        threshold_mid2 = 50.05
        array_prctile_mid1 = np.percentile(array, threshold_mid1)
        array_prctile_mid2 = np.percentile(array, threshold_mid2)
        chosen_timesteps_mid = np.squeeze(np.argwhere((array.data >= array_prctile_mid1) & (array.data <= array_prctile_mid2)))

        threshold_high = 99.9
        threshold_high = 99.5
        rome_prctile = np.percentile(array, threshold_high)
        chosen_timesteps_high= np.squeeze(np.argwhere(array.data>=rome_prctile))

        chosen_timesteps = np.concatenate((chosen_timesteps_low, chosen_timesteps_mid, chosen_timesteps_high))
        chosen_timesteps = chosen_timesteps_high
    return chosen_timesteps


def run_animation(switch):
    keys = [k for k, v in switch.items() if v]                                          # list of True keys
    switch_0, switch_1, switch_t = switch.copy(), switch.copy(), switch.copy() 
    switch_0 =  {k: False if k in keys[1:3] else v for k, v in switch.items()}          # Creates switch for first metric
    switch_1 =  {k: False if k in [keys[0], keys[2]] else v for k, v in switch.items()} #                    second 
    switch_t =  {k: False if k in keys[0:2] else v for k, v in switch.items()}          #                    timestep
    variable_0 = mF.get_variable_object(switch_0)
    variable_1 = mF.get_variable_object(switch_1)
    metric_t   = mF.get_metric_object(switch_t)

    print(f'Creating animation of {variable_1.name} ontop {variable_0.name} on days picked by threshold on {metric_t.option} \n from {mV.resolutions[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    da_0 = get_da(switch, variable_0)
    da_1 = get_da(switch, variable_1)
    timesteps = get_timesteps(switch, metric_t)

    title = f'{metric_t.option}_and_{variable_1.name}_on_{variable_0.name}'
    ani = animate(da_0, da_1, timesteps, variable_0, variable_1, metric_t, title)

    source = mF.find_source(mV.datasets[0], mV.models_cmip5, mV.models_cmip6, mV.observations)
    folder = f'{mV.folder_save[0]}/{metric_t.variable_type}/animations/{source}'
    filename = title
    filename = 'test'

    ani.save(f'{os.getcwd()}/test.mp4')                if switch['save to cwd'] else None
    ani.save(f'{home}/Desktop/{filename}.mp4')         if switch['save to desktop'] else None
    ani.save(f'{folder}/{filename}.mp4')               if switch['save'] else None

 
if __name__ == '__main__':
    run_animation(switch = {
        # -----------------
        # fields to animate (background + field ontop possible)
        # -----------------
            # daily
            'obj':                 True,
            'pr':                  False,
            'pr99':                True,
            'mse':                 False,

            # monthly
            'hur':                 False,
            'rlut':                False,

        # timesteps derived from
        'rome':                True,

        # get data from
        'constructed_fields':  False, 
        'sample_data':         True,
        'gadi_data':           False,

        # masked by
        'fixed area':          False,
        'ascent':              False,
        'descent':             False,

        # type of animation
        'high extremes':       False,
        'low extremes':        False,
        'transition':          False

        # save
        'save':                False,
        'save to cwd':         False,
        'save to desktop':     True
        }
    )















































































