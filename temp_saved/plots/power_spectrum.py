import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import fft
import os
import sys
home = os.path.expanduser("~")                           
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myFuncs as mF     
import myClasses as mC



# ------------------------
#       Get list
# ------------------------
# --------------------------------------------------------------------------------- load list ----------------------------------------------------------------------------------------------------- #
def load_random_data(seed):
    dates = pd.date_range(start='1970-01-01', end='1999-12-31', freq='D')
    np.random.seed(seed)
    data = np.random.normal(size=len(dates))                                    
    return data, dates

def load_data(switchM, dataset, metric_class):
    source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
    timescale = 'daily' if metric_class.var_type in ['pr', 'org', 'hus', 'ws'] else 'monthly'

    experiment  = '' if source == 'obs' else mV.experiments[0]
    # dataset = 'GPCP_1998-2009' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset
    dataset = 'GPCP_2010-2022' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset                                   # pick a time range for obs org
    # dataset = 'GPCP' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset # complete record

    alist = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, experiment, mV.resolutions[0])[metric_class.name]   
    alist = mF.resample_timeMean(alist, mV.timescales[0])
    axtitle = dataset
    
    if dataset == 'CERES':
        alist['time'] = alist['time'] - pd.Timedelta(days=14)

    metric_title = ''
    if switchM['anomalies']:
        metric_title = 'anomalies'
        if mV.timescales[0] == 'daily': 
            rolling_mean = alist.rolling(time=12, center=True).mean()
            alist = alist - rolling_mean
            alist = alist.dropna(dim='time')
        if mV.timescales[0] == 'monthly': 
            climatology = alist.groupby('time.month').mean('time')
            alist = alist.groupby('time.month') - climatology 
        if mV.timescales[0] == 'annual':
            '' 
    return alist, metric_title, axtitle



# ------------------------
#  Calculate plot metric
# ------------------------
# -------------------------------------------------------------------------------- power spectrum ----------------------------------------------------------------------------------------------------- #
def get_powerspectrum(alist, dates):
    frequencies = np.fft.fftfreq(len(dates), d=(dates[1] - dates[0]).days)
    positive_freqs = frequencies > 0
    mjo_range = (1/400 <= frequencies) & (frequencies <= 1/10)                  # check frequencies up to about annual     
    filtered_frequencies = frequencies[positive_freqs & mjo_range]
    periods = 1 / filtered_frequencies                                          # display as period [timefrequency^-1]

    fft_results = fft(alist.data)                                                     
    power = np.abs(fft_results)
    filtered_power = power[positive_freqs & mjo_range]                          # Calculate Fourier Transform
    return periods, filtered_power


# def get_powerspectrum(alist, dates):
#     # Assuming 'dates' is an xarray DataArray with datetime64 type
#     d = dates.diff(dim='time').mean().dt.days  # More intuitive handling of time deltas

#     frequencies = xr.apply_ufunc(np.fft.fftfreq, len(dates), d)
#     positive_freqs = frequencies > 0
#     mjo_range = (1/400 <= frequencies) & (frequencies <= 1/10)  # Check frequencies up to about annual

#     filtered_frequencies = frequencies.where(positive_freqs & mjo_range, drop=True)
#     periods = 1 / filtered_frequencies  # Display as period [timefrequency^-1]

#     fft_results = alist.fft(dim='time')  # Using xarray's built-in FFT method
#     power = np.abs(fft_results)
#     filtered_power = power.where(positive_freqs & mjo_range, drop=True)  # Calculate Fourier Transform

#     return periods, filtered_power


# ------------------------------------------------------------------------------- get plot metric ----------------------------------------------------------------------------------------------------- #
def get_plot_metric(switchM, metric_class, alist, dates):
    if switchM['powerspectrum']:
        title = f'{metric_class.name} powerspectrum'
        xlabel = f'period [{mV.timescales[0]}]'
        ylabel =  'power'
        x, y = get_powerspectrum(alist, dates)
    return x, y, title, xlabel, ylabel



# ------------------------
#        Plot
# ------------------------
def plot_one_lineplot(x, y, title, xlabel, ylabel):
    fig = plt.figure(figsize=(10, 6))
    plt.plot(x, y)                              # Plotting the power spectrum in the MJO frequency range
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    return fig

def plot_multiple_lineplots(switchM, metric_class):
    ncols = 4
    nrows = int(np.ceil(len(mV.datasets)/ ncols))
    width, height = [12, 8.5]   if nrows == 5 else [12, 8.5] 
    width, height = [12, 10]    if nrows == 6 else [width, height]
    width, height = [12, 11.5]  if nrows == 7 else [width, height]
    width, height = [12, 11.5]  if nrows == 8 else [width, height]

    fig, axes = mF.create_figure(width = 12, height = 8.5, nrows=nrows, ncols=ncols)
    num_subplots = len(mV.datasets)
    for i, dataset in enumerate(mV.datasets):
        row = i // ncols  # determine row index
        col = i % ncols   # determine col index
        ax = axes.flatten()[i]
        alist, metric_title, axtitle    = load_data(switchM, dataset, metric_class)
        dates, alist                    = mF.convert_to_datetime(alist.time.values, alist)
        x, y, title, xlabel, ylabel     = get_plot_metric(switchM, metric_class, alist, dates)
        title = f'{title} {metric_title}'

        ax.plot(x, y)
        mF.move_col(ax, -0.0715+0.0025)        if col == 0 else None
        mF.move_col(ax, -0.035)                if col == 1 else None
        mF.move_col(ax, 0.0)                   if col == 2 else None
        mF.move_col(ax, 0.035)                 if col == 3 else None

        mF.move_row(ax, 0.0875 - 0.025 +0.025) if row == 0 else None
        mF.move_row(ax, 0.0495 - 0.0135+0.025) if row == 1 else None
        mF.move_row(ax, 0.01   - 0.005+0.025)  if row == 2 else None
        mF.move_row(ax, -0.0195+0.025)         if row == 3 else None
        mF.move_row(ax, -0.05+0.025)           if row == 4 else None
        mF.move_row(ax, -0.05+0.01)            if row == 5 else None
        mF.move_row(ax, -0.05+0.01)            if row == 6 else None
        mF.move_row(ax, -0.05+0.01)            if row == 7 else None

        mF.scale_ax_x(ax, 0.9)
        mF.scale_ax_y(ax, 0.85)

        mF.plot_xlabel(fig, ax, xlabel, pad=0.055, fontsize = 10)    if i >= num_subplots-ncols else None
        mF.plot_ylabel(fig, ax, ylabel, pad = 0.0475, fontsize = 10) if col == 0 else None
        mF.plot_axtitle(fig, ax, axtitle, xpad = 0.05, ypad = 0.0075, fontsize = 10)
        ax.text(0.5, 0.985, title, ha = 'center', fontsize = 9, transform=fig.transFigure)
        ax.set_xticklabels([]) if not i >= num_subplots-ncols else None
    mF.delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, title




# ------------------------
#     Run / save plot
# ------------------------
# ---------------------------------------------------------------------------------- Find metric / labels and run ----------------------------------------------------------------------------------------------------- #
def plot_lineplot(switchM, switch, metric_class):
    if len(mV.datasets) == 1:  
        alist, metric_title, axtitle    = load_data(switchM, mV.datasets[0], metric_class)
        dates, alist                    = mF.convert_to_datetime(alist.time.values, alist)
        x, y, title, xlabel, ylabel     = get_plot_metric(switchM, metric_class, alist, dates)
        title                           = f'{title} {metric_title} {axtitle}'
        fig                             = plot_one_lineplot(x, y, title, xlabel, ylabel)

    else:
        fig, title = plot_multiple_lineplots(switchM, metric_class)

    mF.save_plot(switch, fig, home, title)
    plt.show() if switch['show'] else None




@mF.timing_decorator
def run_line_plot(switch_metric, switchM, switch):
    print(f'plotting {len(mV.datasets)} datasets with {mV.timescales[0]} {mV.resolutions[0]} data')
    print(f'metric: {[key for key, value in switch_metric.items() if value]}')
    print(f'metric_type: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')

    if switch_metric['random']:
        seeds = [0,1]
        for seed in seeds:
            alist, dates                = load_random_data(seed)
            x, y, title, xlabel, ylabel = get_plot_metric(alist, dates, switchM)
            fig                         = plot_one_lineplot(x, y, title, xlabel, ylabel)
            mF.save_plot(switch, fig, home)
            plt.show() if switch['show'] else None
    
    if not switch_metric['random']:
        for metric in [k for k, v in switch_metric.items() if v]:
            metric_class = mC.get_metric_class(metric, switchM, prctile = mV.conv_percentiles[0])
            plot_lineplot(switchM, switch, metric_class)





# ------------------------
#  Choose what to plot
# ------------------------
if __name__ == '__main__':
# ---------------------------------------------------------------------------------- metric ----------------------------------------------------------------------------------------------------- #
    switch_metric = {
        'random':       False,
        'rome':         True,   
        }

    switchM = {
        'anomalies':    False,
        'timeseries':   False,  'powerspectrum':    True,   # metric type
        }


# ---------------------------------------------------------------------------------- settings ----------------------------------------------------------------------------------------------------- #
    switch = {
        'show':                 False,                                                              # show
        'save_test_desktop':    True,   'save_folder_desktop':  False,  'save_folder_cwd':  False,  # save
        }


# ------------------------------------------------------------------------------------ run ----------------------------------------------------------------------------------------------------- #
    run_line_plot(switch_metric, switchM, switch)




# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import os
    import sys
    home = os.path.expanduser("~")

    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars as mV
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import get_data.metric_data as mD

    dataset = mV.datasets[0]
    experiment = mV.experiments[0]
    metric_type = 'conv_org'
    metric_name = f'rome_{mV.conv_percentiles[0]}thprctile'

    max_length = 0
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, experiment)
        max_length = max(max_length, da.sizes['time'])

    ds = xr.Dataset()
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, experiment)
        current_length = da.sizes['time']
        da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
        if current_length < max_length:
            padding = xr.DataArray(np.full((max_length - current_length,), np.nan), dims=['time'], coords={'time': np.arange(current_length, max_length)})
            da = xr.concat([da, padding], dim='time')
        ds[dataset] = da
    # print(ds)
    # print(ds['ACCESS-CM2'][0:5])
    print(ds['TaiESM1'][-10::])
