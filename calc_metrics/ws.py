import matplotlib.pyplot as plt
import xarray as xr
import os
import sys
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from datetime import datetime, timedelta
import calendar

home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import regrid as rG
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myClasses as mC
import myFuncs as mF



# ------------------------
#       Get data
# ------------------------
# ---------------------------------------------------------------------------------------- load data ----------------------------------------------------------------------------------------------------- #
def load_data(year):
    return xr.open_dataset(f'{home}/Documents/data/sample_data/ws/{str(year)}.nc')



# ------------------------
#   intermediate plot
# ------------------------
# ------------------------------------------------------------------------------- plot intermediate calculation ----------------------------------------------------------------------------------------------------- #
def plot_scene(scene, title = ''):
    fig, ax = mF.create_map_figure(width = 12, height = 4)
    pcm = mF.plot_axScene(ax, scene, cmap = 'Blues')
    mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.075, numbersize = 12, cbar_label = 'weather states frequency [Nb]', text_pad = 0.125)
    mF.move_col(ax, moveby = -0.055)
    mF.move_row(ax, moveby = 0.075)
    mF.scale_ax(ax, scaleby = 1.15)
    mF.plot_xlabel(fig, ax, xlabel='Lon', pad = 0.1, fontsize = 12)
    mF.plot_ylabel(fig, ax, ylabel='Lat', pad = 0.055, fontsize = 12)
    mF.format_ticks(ax, labelsize = 10)
    mF.plot_axtitle(fig, ax, f'ISCCP: {title}', xpad = 0.005, ypad = 0.025, fontsize=15)
    return fig, ax

def plot_scan(switch, da_month, month, year):
    for slice_3hr in da_month[da_month.dims[0]]:
        scene = da_month[slice_3hr, :, :]
        new_lat = np.arange(90, -90, -1)
        scene = xr.DataArray(
            data = np.rot90(scene.data),   
            dims=['lat', 'lon'],
            coords={'lat': new_lat, 'lon': scene.longitude.data}
            )
        scene = xr.where(scene>0, 1, np.nan)
        scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
        fig, _ = plot_scene(switch, scene, title = f'3hr scan: {slice_3hr.data}, month: {month}, year: {year}')
        plt.show()
        mF.save_figure(fig, f'{home}/Desktop/ws', 'ws_scan.pdf') if switch['save_to_desktop'] and year == 2000 and month == 'january' and slice_3hr.data == 0 else None

def plot_daily_scene(switch, metric, scene, year, month, day):
    plot_scene(switch, scene, title = f'{metric} daily freq, day: {day}, month: {month}, year: {year}') 
    plt.show() 

def plot_tMean_scene(switch, metric, scene, year, month):
    plot_scene(switch, scene,  title = f'{metric} average daily freq (tMean cumulative), year:{year}, month:{month}') 
    plt.show()



# ------------------------
#    Calculate metrics
# ------------------------
# -------------------------------------------------------------------------- Field from which metric is calculated ----------------------------------------------------------------------------------------------------- #
def get_snapshot(ws_values, da_month):
    start = 0
    slice_dict = {da_month.dims[0]: slice(start, start + 8)} # picks out sections of 8 slices (1 day)
    scene_day = da_month.isel(**slice_dict)
    slice_dict = {da_month.dims[0]: 0} # picks out first day of month (day dim)
    scene = xr.zeros_like(da_month.isel(**slice_dict)) 
    for ws in ws_values:
        scene += xr.where(scene_day == ws, 1, 0).sum(dim=da_month.dims[0])
    scene = xr.DataArray(data = np.rot90(scene.data), dims=['lat', 'lon'], coords={'lat': np.arange(90, -90, -1), 'lon': scene.longitude.data})
    scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
    return scene


# --------------------------------------------------------------------------------------- spatial mean ----------------------------------------------------------------------------------------------------- #
def calc_daily_freq(switch, metric, ws_values, year, month, da_month):
    plot_scan(switch, da_month, month, year) if switch['show_scans'] else None

    # create time coordinates
    days_in_month = len(range(0, len(da_month[da_month.dims[0]]), 8))
    month_num = list(calendar.month_name).index(month.capitalize())
    start_date = datetime(year, month_num, 1)
    dates = [start_date + timedelta(days=i) for i in range(days_in_month)]

    # calculate sMean
    sMean = []
    for day, start in enumerate(range(0, len(da_month[da_month.dims[0]]), 8)):
        slice_dict = {da_month.dims[0]: slice(start, start + 8)} # picks out sections of 8 slices (1 day)
        scene_day = da_month.isel(**slice_dict)
        slice_dict = {da_month.dims[0]: 0} # picks out first day of month (day dim)
        scene = xr.zeros_like(da_month.isel(**slice_dict)) 
        for ws in ws_values:
            scene += xr.where(scene_day == ws, 1, 0).sum(dim=da_month.dims[0])
        scene = xr.DataArray(data = np.rot90(scene.data), dims=['lat', 'lon'], coords={'lat': np.arange(90, -90, -1), 'lon': scene.longitude.data})
        scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
        plot_daily_scene(switch, metric, scene, year, month, day) if switch['show_daily_scenes'] else None
        sMean_day = scene.weighted(np.cos(np.deg2rad(scene.lat))).mean(dim=('lat','lon'), keep_attrs=True)
        sMean = np.append(sMean, sMean_day)
    return dates, sMean


# ----------------------------------------------------------------------------------------- time mean ----------------------------------------------------------------------------------------------------- #
def calc_tMean_freq(switch, metric, ws_values, year, month, da_month, tMean):
    nb_days = len(da_month[da_month.dims[0]])/8        
    slice_dict = {da_month.dims[0]: 0}
    scene = xr.zeros_like(da_month.isel(**slice_dict)) 
    for ws in ws_values:
        scene += xr.where(da_month == ws, 1, 0).sum(dim=da_month.dims[0]) 
    scene = scene / nb_days
    scene = xr.DataArray(data = np.rot90(scene.data), dims=['lat', 'lon'], coords={'lat': np.arange(90, -90, -1), 'lon': scene.longitude.data})
    scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
    plot_tMean_scene(switch, metric, scene, year, month) if switch['show_tMean_scenes'] else None
    if tMean is None:
        tMean = scene
    else:
        tMean += scene
    return tMean



# ------------------------
#   Run / save metrics
# ------------------------
# -------------------------------------------------------------------------------------- Put into dataset and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, source, years, metric, ws_values, metric_type):
    print(f'{metric} {metric_type}')
    ws_freq_sMean, ws_freq_dates = [], []
    ws_freq_tMean = None    
    for year in years:
        print(f'year: {year} started')
        ds = load_data(year)
        for month in ds.data_vars:
            if month == 'ws':
                continue
            da_month = ds[month] # has dims (3hr_slice, lon, lat)

            if metric_type == 'snapshot':
                metric_name =f'{metric}_snapshot'
                da_calc = get_snapshot(ws_values, da_month)
                break

            if metric_type == 'sMean':
                metric_name =f'{metric}_sMean'
                dates, sMean = calc_daily_freq(switch, metric, ws_values, year, month, da_month) # sMean is a list of daily values for a month here 
                ws_freq_dates.extend(dates)
                ws_freq_sMean= np.append(ws_freq_sMean, sMean)
                da_calc = xr.DataArray(data = ws_freq_sMean, dims = ['time'], coords = {'time': ws_freq_dates})

            if metric_type == 'tMean':
                metric_name =f'{metric}_tMean'
                ws_freq_tMean = calc_tMean_freq(switch, metric, ws_values, year, month, da_month, ws_freq_tMean)
                da_calc = ws_freq_tMean / (len(years)*12) # divide by nb of months

        if metric_type == 'snapshot':
            break                

    mF.save_in_structured_folders(da_calc, f'{mV.folder_save[0]}/metrics', 'ws', metric_name, source, mV.datasets[0], mV.timescales[0], mV.experiments[0], mV.resolutions[0]) if switch['save'] else None



# ------------------------
#    Run / save metric
# ------------------------
# ---------------------------------------------------------------------------------------------- pick dataset ----------------------------------------------------------------------------------------------------- #
def run_metric(switch_metric, switchM, switch):
    source = mV.find_source(mV.datasets[0], mV.models_cmip5, mV.models_cmip6, mV.observations)
    years = np.arange(2000, 2018) # available years: 1983-2017 (first years do not have complete coverage)
    for metric in [k for k, v in switch_metric.items() if v] :
        ws_values = [7, 8, 9, 10]      if metric == 'ws_lc' else [7]       # taken from examples of weather states from ISCCP (clouds below 600 hPa)
        ws_values = [1, 2, 3, 4, 5, 6] if metric == 'ws_hc' else ws_values # taken from examples of weather states from ISCCP (clouds above 600 hPa)

        for metric_type in [k for k, v in switchM.items() if v]:
            get_metric(switch, source, years, metric, ws_values, metric_type)


@mF.timing_decorator
def run_ws_metrics(switch_metric, switchM, switch):
    if not [mV.datasets[0], mV.experiments[0], mV.timescales[0]] == ['ISCCP', '', 'daily']:
        print('can only do daily obs: ISCCP')
    else:
        print(f'Running {os.path.basename(__file__)} with {mV.resolutions[0]} polar orbiting satellite 3hr slice data')
        print(f'metric: {[key for key, value in switch_metric.items() if value]} {[key for key, value in switchM.items() if value]}')
        print(f'settings: {[key for key, value in switch.items() if value]}')
        run_metric(switch_metric, switchM, switch)



# ------------------------
#   Choose what to run
# ------------------------
# --------------------------------------------------------------------------------------------- choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_metric = {
        # metrics (can choose multiple)
        'ws_lc':                       True,
        'ws_hc':                       True,
        }
    
    switchM = {
        # choose type
        'snapshot':                    False,
        'sMean':                       False, 
        'tMean':                       False, 
        }

    switch = {        
        # show
        'show_scans':                  False, 
        'show_daily_scenes':           False, 
        'show_tMean_scenes':           False, 

        # save
        'save':                        False,
        }
    
    run_ws_metrics(switch_metric, switchM, switch)












