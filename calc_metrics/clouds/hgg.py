import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import timeit

import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/functions')
import myVars as mV         # imports common variables
import regrid as rG         # imports regridder
import myFuncs as mF        # imports common operators and plotting
sys.path.insert(0, f'{folder_code}/plotting')
import map_plot_scene as mp # imports functions to plot maps


# ----------------------------------------------------- functions to visualize / calculate metric ----------------------------------------------------------------------------------------------------- #

def convert_to_map_scene(scene, resolution):
    new_lat = np.arange(90, -90, -1)
    scene = xr.DataArray(
        data = np.rot90(scene.data),   
        dims=['lat', 'lon'],
        coords={'lat': new_lat, 'lon': scene.longitude.data}
        )
    scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if resolution == 'regridded' else scene.sel(lat = slice(30,-30)) 
    return scene

def plot_clouds(scene, cmap, cbar_label):
    fig, ax = mp.create_map_figure(width = 12, height = 4)
    pcm = mp.plot_axScene(ax, scene, cmap, vmin = None, vmax = None)

    mp.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.075, numbersize = 12, cbar_label = cbar_label, text_pad = 0.125)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    mp.format_ticks(ax, labelsize = 10)
    return fig, ax

def plot_3hr(da_month, year, month, resolution):
    for slice_3hr in da_month[da_month.dims[0]]:
        scene = da_month[slice_3hr, :, :]
        scene = convert_to_map_scene(scene, resolution)
        scene = xr.where(scene>0, 1, np.nan)

        cmap = 'Reds'
        cbar_label = 'Frequency of occurance in 3hr slice [Nb]'
        fig, ax = plot_clouds(scene, cmap, cbar_label)
        mF.plot_axtitle(fig, ax, f'ISCCP: 3hr slice nb: {slice_3hr.data}, month: {month}, year: {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
        plt.show()
    return

def plot_day(da_month, year, month, resolution):
    for slice_3hr in da_month[da_month.dims[0]]:
        scene = da_month[slice_3hr, :, :]
        scene = convert_to_map_scene(scene, resolution)
        scene = xr.where(scene>0, 1, np.nan)

        cmap = 'Greens'
        cbar_label = 'Frequency of occurance in day [Nb/day]'
        fig, ax = plot_clouds(scene, cmap, cbar_label)
        mF.plot_axtitle(fig, ax, f'ISCCP: 3hr slice nb: {slice_3hr.data}, month: {month}, year: {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
        plt.show()
    return

def plot_month(da_month, ws, year, month, resolution):
    nb_days = len(da_month[da_month.dims[0]])/8
    scene = xr.where(da_month==ws, 1,0).sum(dim=da_month.dims[0])/nb_days
    scene = convert_to_map_scene(scene, resolution)

    cmap = 'Blues'
    cbar_label = 'Frequency of occurance in month [Nb/day]'
    fig, ax = plot_clouds(scene, cmap, cbar_label)
    mF.plot_axtitle(fig, ax, f'ISCCP: {month}, {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
    plt.show()
    return

def plot_year(scene_year, year):
    cmap = 'Greys'
    cbar_label = 'Frequency of occurance in year [Nb/day]'
    fig, ax = plot_clouds(scene_year, cmap, cbar_label)
    mF.plot_axtitle(fig, ax, f'ISCCP: {year}', xpad = 0.005, ypad = 0.025, fontsize=15)
    plt.show()
    return

def calc_sMean(da_month, ws, resolution):
    nb_days = len(da_month[da_month.dims[0]])/8
    scene = xr.where(da_month==ws, 1,0).sum(dim=da_month.dims[0])/nb_days
    scene = convert_to_map_scene(scene, resolution)
    sMean = scene.mean(dim = ('lat', 'lon'))
    return scene, sMean


# ----------------------------------------------------- Get the data from the dataset / experiment and run ----------------------------------------------------------------------------------------------------- #

def calc_metrics(switch, da_month, ws, year, month, resolution):
    plot_3hr(da_month, year, month, resolution) if switch['3hr'] else None
    plot_day(da_month, ws, year, month, resolution) if switch['day'] else None
    plot_month(da_month, ws, year, month, resolution) if switch['month'] else None
    return calc_sMean(da_month, ws, resolution)


def run_year(switch, ds, ws, year, resolution):
    ws_weighted_freq = []
    scene_year = None
    for month in ds.data_vars:
        if month == 'ws':
            continue
        da_month = ds[month] # has dims (3hr_slice, lon, lat)
        scene, sMean = calc_metrics(switch, da_month, ws, year, month, resolution)
        ws_weighted_freq = np.append(ws_weighted_freq, sMean)

        if scene_year is None:
            scene_year = scene
        else:
            scene_year += scene

    plot_year(scene_year, year) if switch['year'] else None
    return scene_year, ws_weighted_freq 


def run_hgg_metrics(switch, ws, year_start = 1983, year_finish = 2017, folder_save = f'{mV.folder_save}/hgg', resolution = 'regridded'):
    dataset = 'ISCCP'
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    experiment = ''
    print(f'Running ws_metrics with {resolution} 3hrly data')
    print(f'switch: {[key for key, value in switch.items() if value]}')
    print(f'{dataset} ({source})')

    years = np.arange(year_start, year_finish)
    ws_freq = []
    scene_tMean = None
    for year in years:
        ds = xr.open_dataset(f'{home}/Documents/data/hgg/sample_data/{str(year)}.nc')
        scene_year, ws_weighted_freq  = run_year(switch, ds, ws, year, resolution)
        ws_freq = np.append(ws_freq, ws_weighted_freq )

        if scene_tMean is None:
            scene_tMean = scene_year
        else:
            scene_tMean += scene_year
    
    if switch['ws_tMean']:
        ds_ws_tMean = xr.Dataset(data_vars = {'ws_tMean': scene_tMean}, attrs = {'Description': f'weather state (ws) {ws}'})    
        mV.save_metric(ds_ws_tMean, folder_save, 'ws_tMean', source, dataset, experiment, resolution) if switch['save'] else None

    if switch['ws_sMean']:
        ds_ws_sMean = xr.Dataset(data_vars = {'ws_sMean': ws_freq}, attrs = {'Description': f'weather state (ws) {ws}'})    
        mV.save_metric(ds_ws_sMean, folder_save, 'ws_sMean', source, dataset, experiment, resolution) if switch['save'] else None
    return




# -------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #

if __name__ == '__main__':

    start = timeit.default_timer()
    # choose which metrics to plot
    switch = {
        '3hr':      False,
        'day':      False,
        'month':    False,
        'year':     False,
        
        'ws_tMean': False, 
        'ws_sMean': False, 

        'show':     True,
        'save':     False
        }
    
    # Choose weather state (ws) (number between 1-11), and years to plot [1983, 2017]
    run_hgg_metrics(switch, 
                    ws=7, 
                    year_start=1999, 
                    year_finish=2018, 
                    folder_save = f'{mV.folder_save}/hgg') 

    stop = timeit.default_timer()
    print(f'Finshed, script finished in {round((stop-start)/60, 2)} minutes.')
















