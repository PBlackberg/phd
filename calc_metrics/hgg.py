import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/plotting')
import map_plot_scene as mp


# ----------------------------------------------------- functions to visualize / calculate metric ----------------------------------------------------------------------------------------------------- #

def convert_to_map_scene(scene):
    new_lat = np.arange(90, -90, -1)
    scene = xr.DataArray(
        data = np.rot90(scene.data),   
        dims=['lat', 'lon'],
        coords={'lat': new_lat, 'lon': scene.longitude.data}
        )
    return scene.sel(lat = slice(30,-30))

def plot_clouds(scene, cmap, cbar_label):
    fig, ax = mp.create_map_figure(width = 12, height = 4)
    pcm = mp.plot_axScene(ax, scene, cmap, vmin = None, vmax = None)

    mp.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.075, numbersize = 12, cbar_label = cbar_label, text_pad = 0.125)
    mp.move_col(ax, moveby = -0.055)
    mp.move_row(ax, moveby = 0.075)
    mp.scale_ax(ax, scaleby = 1.15)
    mp.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    mp.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    mp.format_ticks(ax, labelsize = 10)
    return fig, ax

def plot_3hr(da_month, month, year):
    for slice_3hr in da_month[da_month.dims[0]]:
        scene = da_month[slice_3hr, :, :]
        scene = convert_to_map_scene(scene)
        scene = xr.where(scene>0, 1, np.nan)
        # print(np.unique(scene))
        cmap = 'Reds'
        cbar_label = 'Frequency of occurance in day'
        fig, ax = plot_clouds(scene, cmap, cbar_label)
        mp.plot_axtitle(fig, ax, f'ISCCP: 3hr slice nb: {slice_3hr.data}, month: {month}, year: {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
        plt.show()
    return

def plot_month(scene, month, year):
    cmap = 'Blues'
    cbar_label = 'Frequency of occurance in month [Nb/day]'
    fig, ax = plot_clouds(scene, cmap, cbar_label)
    mp.plot_axtitle(fig, ax, f'ISCCP: {month}, {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
    plt.show()
    return

# def plot_year(ds, year):
#     total_scene = None

#     for month in ds.data_vars:
#         if month == 'ws':
#             continue
#         da_month = ds[month] # has dims (3hr_slice, lon, lat)
#                     if total_scene is None:
#                     total_scene = scene
#                 else:
#                     total_scene += scene
#             cmap = 'Greys'
#         cbar_label = 'Frequency of occurance in month [Nb/day]'
#         fig, ax = plot_clouds(scene, cmap, cbar_label)
#         mp.plot_axtitle(fig, ax, f'ISCCP: {month}, {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
#         plt.show()
#     return




# ----------------------------------------------------- Get the data from the dataset / experiment and run ----------------------------------------------------------------------------------------------------- #

def run_weather_state(switch, ds, ws, year):
    for month in ds.data_vars:
        if month == 'ws':
            continue
        da_month = ds[month] # has dims (3hr_slice, lon, lat)

        plot_3hr(da_month, month, year) if switch['3hr'] and switch['show'] else None
        # plot_day(scene, month, year) if switch['day'] and switch['show'] else None

        nb_days = len(da_month[da_month.dims[0]])/8
        scene = xr.where(da_month==ws, 1,0).sum(dim=da_month.dims[0])/nb_days
        scene = convert_to_map_scene(scene)
        plot_month(scene, month, year) if switch['month'] and switch['show'] else None

    # plot_year(ds, year)
    
    return scene.mean(dim = ('lat', 'lon'))

def run_hgg_visualization(switch, ws, year_start=1983, year_finish=2017):
    years = np.arange(year_start, year_finish)
    ws_weighted_freq = []
    for year in years:
        ds = xr.open_dataset(f'/Users/cbla0002/Documents/data/hgg/global/{str(year)}.nc')
        ws_weighted_freq = np.append(ws_weighted_freq, run_weather_state(switch, ds, ws, year))

    return ws_weighted_freq







# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    start = timeit.default_timer()
    # choose which metrics to plot
    switch = {
        '3hr'   : True,
        'day'   : False,
        'month' : False,
        'year'  : False,
        
        'show'  : True,
        }
    
    # Choose weather state (ws) (number between 1-11), and years to plot [1983, 2017]
    run_hgg_visualization(switch, ws=7, year_start=1996, year_finish=1997) 

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')








# if finding annual scenes
# total_scene = None
        # if total_scene is None:
        #     total_scene = scene
        # else:
        #     total_scene += scene

















