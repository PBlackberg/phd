
import matplotlib.pyplot as plt
import xarray as xr
import os
import sys
import numpy as np
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)

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



# ------------------------------------------------------------------------------- plot intermediate calculation ----------------------------------------------------------------------------------------------------- #
def plot_scene(switch, scene, title = ''):
    fig, ax = None, None
    if switch['show']:
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
        plt.show()
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
        mF.save_figure(fig, f'{home}/Desktop/ws', 'ws_scan.pdf') if switch['save_example'] and year == 2000 and month == 'january' and slice_3hr.data == 0 else None



# ------------------------
#    Calculate metrics
# ------------------------
# --------------------------------------------------------------------------------------- spatial mean ----------------------------------------------------------------------------------------------------- #
def calc_daily_freq(switch, da_month, month, year, ws_values):
    sMean = []
    for day, start in enumerate(range(0, len(da_month[da_month.dims[0]]), 8)):
        slice_dict = {da_month.dims[0]: slice(start, start + 8)}
        scene_day = da_month.isel(**slice_dict)

        slice_dict = {da_month.dims[0]: 0}
        scene = xr.zeros_like(da_month.isel(**slice_dict)) 
        for ws in ws_values:
            scene += xr.where(scene_day == ws, 1, 0).sum(dim=da_month.dims[0])
        scene = xr.DataArray(data = np.rot90(scene.data), 
                            dims=['lat', 'lon'], 
                            coords={'lat': np.arange(90, -90, -1), 'lon': scene.longitude.data})
        scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
        sMean_day = scene.weighted(np.cos(np.deg2rad(scene.lat))).mean(dim=('lat','lon'), keep_attrs=True)
        sMean = np.append(sMean, sMean_day)
        fig, _ = plot_scene(switch, scene, title = f'ws daily freq, day: {day}, month: {month}, year: {year}')
        mF.save_figure(fig, f'{home}/Desktop/ws', f'ws_daily{mC.c_type(switch)}.pdf') if switch['save_example'] and year == 2000 and month == 'january' and day == 0 else None
    return sMean


def calc_month_freq(switch, da_month, month, year, ws_values):
    nb_days = len(da_month[da_month.dims[0]])/8    
    slice_dict = {da_month.dims[0]: 0}
    scene = xr.zeros_like(da_month.isel(**slice_dict)) 
    for ws in ws_values:
        scene += xr.where(da_month == ws, 1, 0).sum(dim=da_month.dims[0])
    scene = scene / nb_days                                                   # months have different lenght, so frequency expressed in number per day
    scene = xr.DataArray(data = np.rot90(scene.data), 
                        dims=['lat', 'lon'], 
                        coords={'lat': np.arange(90, -90, -1), 'lon': scene.longitude.data})
    scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
    sMean = scene.weighted(np.cos(np.deg2rad(scene.lat))).mean(dim=('lat','lon'), keep_attrs=True)
    fig, _ = plot_scene(switch, scene, title = f'ws month freq, month: {month}, year: {year}')
    mF.save_figure(fig, f'{home}/Desktop/ws', f'ws_monthly{mC.c_type(switch)}.pdf') if switch['save_example'] and year == 2000 and month == 'january' else None
    return sMean


def calc_tMean_freq(da_month, ws_freq_tMean, ws_values):
    nb_days = len(da_month[da_month.dims[0]])/8        
    slice_dict = {da_month.dims[0]: 0}
    scene = xr.zeros_like(da_month.isel(**slice_dict)) 
    for ws in ws_values:
        scene += xr.where(da_month == ws, 1, 0).sum(dim=da_month.dims[0]) 
    scene = scene / nb_days
    scene = xr.DataArray(data = np.rot90(scene.data), 
                        dims=['lat', 'lon'], 
                        coords={'lat': np.arange(90, -90, -1), 'lon': scene.longitude.data})
    scene = rG.regrid_conserv(scene.sel(lat = slice(35,-35))) if mV.resolutions[0] == 'regridded' else scene.sel(lat = slice(30,-30)) 
    if ws_freq_tMean is None:
        ws_freq_tMean = scene
    else:
        ws_freq_tMean += scene
    return ws_freq_tMean



# ------------------------
#   Run / save metrics
# ------------------------
# -------------------------------------------------------------------------------------- Put into dataset and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, metric):
    years = np.arange(2000, 2018)                                               # available: 1983-2017
    ws_values = [7, 8, 9, 10] if switch['low_clouds'] else [7]                  # taken from examples of weather states from ISCCP (clouds below 600 hPa)
    ws_values = [1, 2, 3, 4, 5, 6] if switch['high_clouds'] else ws_values      # taken from examples of weather states from ISCCP (clouds above 600 hPa)

    ws_freq_sMean = []
    ws_freq_tMean = None    
    for year in years:
        ds = load_data(year)
        for month in ds.data_vars:
            if month == 'ws':
                continue
            da_month = ds[month] # has dims (3hr_slice, lon, lat)
            plot_scan(switch, da_month, month, year) if switch['scan'] else None

            if metric == 'sMean':
                metric_name =f'ws{mC.c_type(switch)}_sMean' 
                sMean = calc_daily_freq(switch, da_month, month, year, ws_values) if mV.timescales[0] == 'daily' else None    # sMean is a list here 
                sMean = calc_month_freq(switch, da_month, month, year, ws_values) if mV.timescales[0] == 'monthly' else sMean # sMean is a number here
                ws_freq_sMean = np.append(ws_freq_sMean, sMean)
                da_calc = ws_freq_sMean

            if metric == 'tMean':
                metric_name =f'ws{mC.c_type(switch)}_tMean' 
                ws_freq_tMean = calc_tMean_freq(da_month, ws_freq_tMean, ws_values) if mV.timescales[0] == 'monthly' else None
                da_calc = ws_freq_tMean / (len(years)*12)

        # plot_scene(switch, ws_freq_tMean,  title = f'ws year freq, year: {year}') if mV.timescales[0] == 'monthly' and switch['tMean'] else [None, None]
    fig, _ = plot_scene(switch, ws_freq_tMean,  title = f'ws freq time mean') if mV.timescales[0] == 'monthly' and switch['tMean'] else [None, None ]
    # mF.save_figure(fig, f'{home}/Desktop/ws', f'ws_{mC.c_type(switch)}_tMean.pdf') if switch['save_example'] and year == 2000 and month == 'january' else None
    mF.save_figure(fig, f'{home}/Desktop/ws', f'ws_{mC.c_type(switch)}_tMean.pdf') if switch['save_example'] else None
    mF.save_file(xr.Dataset(data_vars = {metric_name: da_calc}), f'{home}/Desktop/ws/', f'{mV.datasets[0]}_{metric_name}_{mV.timescales[0]}_{mV.experiments[0]}_{mV.resolutions[0]}.nc')



# ---------------------------------------------------------------------------------------------------- pick dataset ----------------------------------------------------------------------------------------------------- #
def run_metric(switch):
    if [mV.datasets[0], mV.experiments[0]] == ['ISCCP', '']:
        for metric in [k for k, v in switch.items() if v] : # loop over true keys
            if metric in ['scan', 'sMean', 'tMean']:
                get_metric(switch, metric)
    else:
        print('can only do obs: ISCCP')

@mF.timing_decorator
def run_ws_metrics(switch):
    print(f'Running {os.path.basename(__file__)} with {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'switch: {[key for key, value in switch.items() if value]}')    
    run_metric(switch)



# --------------------------------------------------------------------------------------------- choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    run_ws_metrics(switch = {
        # visualization
        'scan':                        False, 

        # choose type (up to one)
        'low_clouds':                  False,
        'high_clouds':                 True,

        # choose metrics
        'sMean':                       True, 
        'tMean':                       False, 

        # show / save
        'show':                        True, # plots all timesteps
        'save_example':                False, # plots all timesteps
        'save_to_desktop':             False  # does not plot
        }
    )












