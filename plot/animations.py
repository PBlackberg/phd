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

def plot_ax_scene(frame, fig, switch, da_0, da_1, timesteps, variable_0, variable_1, title):
    lat, lon = da_0.lat, da_0.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax = fig.add_subplot(projection=ccrs.PlateCarree(central_longitude=180))
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    timestep = timesteps[frame]
    pcm_0 = ax.pcolormesh(lonm,latm, da_0.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_0.cmap, vmin=0, vmax=80)
    pcm_1 = ax.pcolormesh(lonm,latm, da_1.isel(time = timestep), transform=ccrs.PlateCarree(),zorder=0, cmap = variable_1.cmap, vmin=0, vmax=80) if switch['field ontop'] else None
    mF.scale_ax(ax, scaleby = 1.15)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)

    if frame == 0:
        mF.plot_axtitle(fig, ax, title, xpad = 0.005, ypad = 0.035, fontsize=15)
        mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
        mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
        mF.format_ticks(ax, labelsize = 11)
        mF.cbar_below_axis(fig, ax, pcm_1, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = variable_1.label, text_pad = 0.125) if switch['field ontop'] \
            else mF.cbar_below_axis(fig, ax, pcm_0, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = variable_0.label, text_pad = 0.125) 
    plt.close()

def animate(switch, da_0, da_1, timesteps, variable_0, variable_1, title):
    fig= plt.figure(figsize=(12, 4))
    ani = animation.FuncAnimation(
        fig,                          
        plot_ax_scene,                                                               # name of the function
        frames = len(timesteps),                                                     # can also be iterable or list
        interval = 500,                                                              # ms between frames
        fargs=(fig, switch, da_0, da_1, timesteps, variable_0, variable_1, title)    # additional function arguments
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
    da = xr.open_dataset(f'{mV.folder_save[0]}/{variable.variable_type}/sample_data/{source}/{mV.datasets[0]}_{variable.name}_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')['precip'] if switch['sample_data'] else da #variable.name
    da = gD.get_pr(source, mV.datasets[0], mV.timescales[0], mV.experiments[0], mV.resolutions[0])                      if switch['gadi_data'] else da
    return da

def calc_vertical_mean(da):
    da = da.sel(plev=slice(850e2, 0)) # free troposphere (most values at 1000 hPa over land are NaN)
    return (da * da.plev).sum(dim='plev') / da.plev.sum(dim='plev')

def get_da(switch, variable):
    if variable.ref == 'obj':
        da = load_data(switch, variable)
        conv_threshold = calc_conv_threshold(da, conv_percentile = int(mV.conv_percentiles[0]) * 0.01, fixed_area = switch['fixed area'])
        da = da.where(da >= conv_threshold)

    if variable.ref == 'pr99':
        da = load_data(switch, variable)
        conv_threshold = calc_conv_threshold(da, conv_percentile = 0.99, fixed_area = True)
        da = da.where(da >= conv_threshold)

    if variable.ref in ['rlut']:
        da = load_data(switch, variable)

    if variable.ref == 'hur':
        da = load_data(switch, variable)
        da = calc_vertical_mean(da)
    return da

def load_array(metric_t):
    timescale = 'daily'
    source = mF.find_source(mV.datasets[0], mV.models_cmip5, mV.models_cmip6, mV.observations)
    array = xr.open_dataset(f'{mV.folder_save[0]}/{metric_t.variable_type}/metrics/{metric_t.name}/{source}/{mV.datasets[0]}_{metric_t.name}_{mV.conv_percentiles[0]}thPrctile_{timescale}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')[metric_t.option]
    array = mF.resample_timeMean(array, mV.timescales[0])
    return array
    
def get_timesteps(switch, metric_t):
    array = load_array(metric_t)
    low, mid_1, mid_2, high = 0.5, 49.975, 50.025, 99.5     # for daily
    low, mid_1, mid_2, high = 5, 47.5, 52.5, 95             # for monthly
    low, mid_1, mid_2, high = 1, 49.5, 50.5, 99             # for daily obs
    timesteps_low  = np.squeeze(np.argwhere(array.data  <= np.percentile(array, low)))
    timesteps_mid  = np.squeeze(np.argwhere((array.data >= np.percentile(array, mid_1)) & (array.data <= np.percentile(array, mid_2))))
    timesteps_high = np.squeeze(np.argwhere(array.data  >= np.percentile(array, high)))
    if switch['low extremes']:
        title = '_low_extremes'
        return timesteps_low, title
    if switch['high extremes']:
        title = '_high_extremes'
        return timesteps_high, title
    if switch['transition']:
        title = '_transition'
        return np.concatenate((timesteps_low, timesteps_mid, timesteps_high)), title


@mF.timing_decorator
def run_animation(switch):
    keys = [k for k, v in switch.items() if v]                                          # list of True keys
    switch_0, switch_1, switch_t = switch.copy(), switch.copy(), switch.copy() 
    switch_0 =  {k: False if k in keys[1:3] else v for k, v in switch.items()}          # Creates switch for first metric
    switch_1 =  {k: False if k in [keys[0], keys[2]] else v for k, v in switch.items()} #                    second 
    switch_t =  {k: False if k in keys[0:2] else v for k, v in switch.items()}          #                    timestep
    variable_0 = mF.get_variable_object(switch_0)
    variable_1 = mF.get_variable_object(switch_1)
    metric_t   = mF.get_metric_object(switch_t)

    print(f'Creating animation of {variable_0.name} on days picked by threshold on {metric_t.option} \n from {mV.resolutions[0]} data') if not switch['field ontop'] \
        else print(f'Creating animation of {variable_1.name} ontop of {variable_1.name} on days picked by threshold on {metric_t.option} \n from {mV.resolutions[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')

    da_0 = get_da(switch, variable_0)
    da_1 = get_da(switch, variable_1)
    timesteps, title = get_timesteps(switch, metric_t)
    print('number of timesteps for animation:', len(timesteps))
    title = f'{mV.datasets[0]}_{metric_t.option}{title}_and_{variable_1.ref}_with_{variable_0.ref}_{mV.timescales[0]}' if switch['field ontop'] else f'{mV.datasets[0]}_{metric_t.option}{title}_and_{variable_0.ref}_{mV.timescales[0]}'

    ani = animate(switch, da_0, da_1, timesteps, variable_0, variable_1, title)

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
        'field ontop':         True,
        'low extremes':        False,
        'high extremes':       False,
        'transition':          True,

        # save
        'save to cwd':         False,
        'save to desktop':     True,
        'save':                False
        }
    )








