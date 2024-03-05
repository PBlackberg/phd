''' 
# ------------------------
#     Scatter plot
# ------------------------
This script plots scatter plots with trend line and correlation coefficient

to use:
sys.path.insert(0, f'{os.getcwd()}/util-plot')
import scatter_time as sT
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors




# --------------------------
#   Plot correlation plot
# --------------------------
# ---------------------------------------------------------------------------------------- General --------------------------------------------------------------------------------------------------- #
def create_figure(width = 12, height = 4, nrows = 1, ncols = 1):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height))
    return fig, axes

def delete_remaining_axes(fig, axes, num_subplots, nrows, ncols):
    for i in range(num_subplots, nrows * ncols):
        fig.delaxes(axes.flatten()[i])

def move_col(ax, moveby):
    ax_position = ax.get_position()
    _, bottom, width, height = ax_position.bounds
    new_left = _ + moveby
    ax.set_position([new_left, bottom, width, height])

def move_row(ax, moveby):
    ax_position = ax.get_position()
    left, _, width, height = ax_position.bounds
    new_bottom = _ + moveby
    ax.set_position([left, new_bottom, width, height])

def scale_ax_x(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2
    ax.set_position([left, bottom, new_width, new_height])

def scale_ax_y(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def plot_xlabel(fig, ax, xlabel, pad, fontsize):
    ax_position = ax.get_position()
    lon_text_x =  ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
    lon_text_y =  ax_position.y0 - pad
    ax.text(lon_text_x, lon_text_y, xlabel, ha = 'center', fontsize = fontsize, transform=fig.transFigure)

def plot_ylabel(fig, ax, ylabel, pad, fontsize):
    ax_position = ax.get_position()
    lat_text_x = ax_position.x0 - pad
    lat_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2
    ax.text(lat_text_x, lat_text_y, ylabel, va = 'center', rotation='vertical', fontsize = fontsize, transform=fig.transFigure)

def plot_axtitle(fig, ax, title, xpad, ypad, fontsize):
    ax_position = ax.get_position()
    title_text_x = ax_position.x0 + xpad 
    title_text_y = ax_position.y1 + ypad
    ax.text(title_text_x, title_text_y, title, fontsize = fontsize, transform=fig.transFigure)

def cbar_right_of_axis(fig, ax, h, width_frac, height_frac, pad, numbersize = 8, cbar_label = '', text_pad = 0.1, fontsize = 10):
    # colorbar position
    ax_position = ax.get_position()
    cbar_bottom = ax_position.y0
    cbar_left = ax_position.x1 + pad
    cbar_width = ax_position.width * width_frac
    cbar_height = ax_position.height * height_frac
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(h, cax=cbar_ax, orientation='vertical')
    cbar.ax.tick_params(labelsize=numbersize)
    # colobar label
    cbar_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2
    cbar_text_x = cbar_left + cbar_width + text_pad
    ax.text(cbar_text_x, cbar_text_y, cbar_label, rotation = 'vertical', va = 'center', fontsize = fontsize, transform=fig.transFigure)
    return cbar

def cbar_below_axis(fig, ax, h, pad = 0.1, numbersize = 8, cbar_label = 'test label [units]', text_pad = 0.05):
    # colorbar position
    cbar_left, cbar_bottom, cbar_width, cbar_height = 0.565, 0.075, 0.3, 0.025
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    ax_position = cbar_ax.get_position()
    cbar = fig.colorbar(h, cax=cbar_ax, orientation='horizontal')
    # colobar label
    cbar.ax.tick_params(labelsize=numbersize)
    cbar_text_x = ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
    cbar_text_y = cbar_bottom - text_pad
    ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 10, transform=fig.transFigure)
    return cbar

def highlight_subplot_frame(ax, color):
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(2)  # Adjust line width

def add_correlation_coeff(x, y, ax):
    res= stats.pearsonr(x,y)
    placement = (0.675, 0.05) if res[0]>0 else (0.675, 0.85) 
    ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext = placement, textcoords='axes fraction', fontsize = 8, color = 'r') if res[1]<=0.05 else None

def xy_align(x, y):
    x, y = x.dropna(dim='time', how='any'), y.dropna(dim='time', how='any')
    x, y = xr.align(x, y, join='inner')
    return x, y

def pad_length_difference(da, max_length = 2): # max_length = 11000
    ''' When creating time series metrics, differnet dataset will have different lenghts.
        To put metrics from all datasets into one xarray Dataset the metric is padded with nan '''
    current_length = da.sizes['time']
    da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
    if current_length < max_length:
        padding = xr.DataArray(np.full((max_length - current_length,), np.nan), dims=['time'], coords={'time': np.arange(current_length, max_length)})
        da = xr.concat([da, padding], dim='time')
    return da

# ---------------------------------------------------------------------------------------- specific --------------------------------------------------------------------------------------------------- #
def format_fig(num_subplots, fig_title):
    ncols = 4 if num_subplots > 4 else num_subplots # maximum 4 subplots per row
    nrows = int(np.ceil(num_subplots / ncols))
    width, height = [12, 8.5]   if nrows == 5 else [12, 8.5] 
    width, height = [12, 10]    if nrows == 6 else [width, height]
    width, height = [12, 11.5]  if nrows == 7 else [width, height]
    width, height = [12, 11.5]  if nrows == 8 else [width, height]
    ncols = 4 if num_subplots > 4 else num_subplots # max 4 subplots per row
    fig, axes = create_figure(width = 12, height = 8.5, nrows=nrows, ncols=ncols)
    delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    fig.text(0.5, 0.985, fig_title, ha = 'center', fontsize = 9, transform=fig.transFigure)
    return fig, axes, nrows, ncols

def format_axes(fig, ax, nrows, ncols, subplot, num_subplots, xmin, xmax, ymin, ymax, density_map, label_x, label_y, variable, models_highlight):
    row, col = subplot // ncols, subplot % ncols    # determine row and col index
    move_col(ax, -0.0715+0.0025)        if col == 0 else None
    move_col(ax, -0.035)                if col == 1 else None
    move_col(ax, 0.0)                   if col == 2 else None
    move_col(ax, 0.035)                 if col == 3 else None

    move_row(ax, 0.0875 - 0.025 +0.025) if row == 0 else None
    move_row(ax, 0.0495 - 0.0135+0.025) if row == 1 else None
    move_row(ax, 0.01   - 0.005+0.025)  if row == 2 else None
    move_row(ax, -0.0195+0.025)         if row == 3 else None
    move_row(ax, -0.05+0.025)           if row == 4 else None
    move_row(ax, -0.05+0.01)            if row == 5 else None
    move_row(ax, -0.05+0.01)            if row == 6 else None
    move_row(ax, -0.05+0.01)            if row == 7 else None

    scale_ax_x(ax, 0.9) # 0.95
    scale_ax_y(ax, 0.85)
    plot_xlabel(fig, ax, label_x, pad=0.055, fontsize = 10)    if subplot >= num_subplots-ncols else None
    plot_ylabel(fig, ax, label_y, pad = 0.0475, fontsize = 10) if col == 0 else None
    plot_axtitle(fig, ax, variable, xpad = 0.035, ypad = 0.0075, fontsize = 10)
    ax.set_xticklabels([]) if not subplot >= num_subplots-ncols else None
    highlight_subplot_frame(ax, color = 'b') if variable in models_highlight else None
    # highlight_subplot_frame(ax, color = 'g') if variable in mV.observations  else None

def plot_axScatter(x, y, ax, color, xmin, xmax, ymin, ymax):
    hs = ax.scatter(x, y, facecolors='none', edgecolor= color) 
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    ax.xaxis.set_major_formatter(formatter)
    return hs

def plot_slope(x, y, ax, color):
    ''' Line of best fit (linear) '''
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = intercept + slope * x
    ax.plot(x, y_fit, color=color)

def add_density_map(x, y, fig, ax, cmap, subplot, ncols, vmin, vmax, individual_cmap):
    row, col = subplot // ncols, subplot % ncols    # determine row and col index
    # vmin = 0
    # vmax = 150
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    hm = ax.hist2d(x,y,[20,20], range=[[np.nanmin(x), np.nanmax(x)], [np.nanmin(y), np.nanmax(y)]], cmap = cmap, norm = norm) 
    cbar_label = 'months [Nb]' if col == 3 else ''
    text_pad = 0.035           if col == 3 else 0.1
    cbar_right_of_axis(fig, ax, hm[3], width_frac= 0.05, height_frac=1, pad=0.015, numbersize = 9, cbar_label = cbar_label, text_pad = text_pad) if individual_cmap else None
    return hm

def plot_scatter(x, y, variable, fig_title, label_x, label_y, colors_scatter, colors_slope, xmin, xmax, ymin, ymax, vmin, vmax, subplot = 0):
    fig, ax = create_figure(width = 9, height = 6)
    x, y = xy_align(x, y)
    h = plot_axScatter(x, y, ax, colors_scatter[subplot], xmin = None, xmax = None, ymin = None, ymax = None)
    hs = plot_slope(x, y, ax, colors_slope[subplot]) 
    move_col(ax, -0.03)
    move_row(ax, 0.01)
    scale_ax_x(ax, 0.95)
    scale_ax_y(ax, 1)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12) 
    individual_cmap = True
    cmap = 'Blues'
    ncols = 1
    hm = add_density_map(x, y, fig, ax, cmap, subplot, ncols, vmin, vmax, individual_cmap)
    add_correlation_coeff(x, y, ax)
    # h = plot_axScatter(x, y, x)
    # res= stats.pearsonr(x,y)
    # placement = (0.825, 0.05) if res[0]>0 else (0.825, 0.9) 
    # ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=placement, textcoords='axes fraction', fontsize = 12, color = 'r') if res[1]<=0.05 else None      
    # ax.set_xlim(xmin, xmax)
    # ax.set_ylim(ymin, ymax)
    # formatter = ticker.ScalarFormatter(useMathText=True)
    # formatter.set_scientific(True)
    # formatter.set_powerlimits((-1,1))
    # ax.xaxis.set_major_formatter(formatter)
    plot_xlabel(fig, ax, label_x, pad = 0.09,   fontsize = 12)
    plot_ylabel(fig, ax, label_y, pad = 0.075, fontsize = 12)
    plot_axtitle(fig, ax, variable, xpad = 0.005, ypad = 0.01, fontsize = 12)
    fig.text(0.5, 0.95, fig_title, ha = 'center', fontsize = 15.5, transform=fig.transFigure)
    cbar_right_of_axis(fig, ax, hm[3], width_frac= 0.05, height_frac=1, pad=0.035, numbersize = 12, cbar_label = 'months [Nb]', text_pad = 0.05, fontsize = 12)
    return fig, ax

def plot_dsScatter(ds_x, ds_y, variable_list = ['a', 'b'], fig_title = '', label_x = '', label_y = '', label_cmap = '',
                    colors_scatter = 'k', colors_slope = 'k', cmap = 'Blues',
                    xmin = None, xmax = None, ymin = None, ymax = None, vmin = 0, vmax = 150,
                    density_map = False, models_highlight = ['a', 'b'],
                    fig_given = False, ax = '',
                    individual_cmap = False):
    
    if len(variable_list) == 1:
        for variable in variable_list:
            return plot_scatter(ds_x[variable], ds_y[variable], variable, fig_title, label_x, label_y, colors_scatter, colors_scatter, xmin, xmax, ymin, ymax, vmin, vmax)
    fig, axes, nrows, ncols = format_fig(len(variable_list), fig_title)
    for subplot, variable in enumerate(variable_list):
        x, y = ds_x[variable], ds_y[variable]
        x, y = xy_align(x, y)
        ax = axes.flatten()[subplot]
        h = plot_axScatter(x, y, ax, colors_scatter[subplot], xmin = None, xmax = None, ymin = None, ymax = None)
        hs = plot_slope(x, y, ax, colors_slope[subplot]) 
        hm = add_density_map(x, y, fig, ax, cmap, subplot, ncols, vmin, vmax, individual_cmap)
        format_axes(fig, ax, nrows, ncols, subplot, len(variable_list), xmin, xmax, ymin, ymax, density_map, label_x, label_y, variable, models_highlight)
        add_correlation_coeff(x, y, ax)
        # cbar_height  = 0.1
        # pad = 0.1
    cbar_below_axis(fig, ax, hm[3], cbar_label = label_cmap) if not individual_cmap else None # cbar_height, pad, numbersize = 8, cbar_label = '', text_pad = 0.1
    return fig, axes



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
    import os
    import sys
    home = os.path.expanduser("~")
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars as mV
    import myFuncs_plots        as mFp
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import get_data.metric_data as mD


# ----------------------------------------------------------------------------------------- get data --------------------------------------------------------------------------------------------------- #
    experiment = mV.experiments[0]
    metric_type = 'conv_org'
    metric_name = f'rome_{mV.conv_percentiles[0]}thprctile'

    max_length = 0
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, experiment)
        max_length = max(max_length, da.sizes['time'])

    ds_x = xr.Dataset()
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, mV.experiments[0])
        current_length = da.sizes['time']
        da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
        if current_length < max_length:
            padding = xr.DataArray(np.full((max_length - current_length,), np.nan), dims=['time'], coords={'time': np.arange(current_length, max_length)})
            da = xr.concat([da, padding], dim='time')
        ds_x[dataset] = da
    # print(ds_x)
    # exit()
    xmin, xmax = None, None
    for var_name, data_array in ds_x.data_vars.items():
        min_var = data_array.min().values
        max_var = data_array.max().values
        if xmin is None or min_var < xmin:
            xmin = min_var
        if xmax is None or max_var > xmax:
            xmax = max_var
    # print(xmin, xmax)

    metric_type = 'pr'
    metric_name = f'pr_sMean'
    ds_y = xr.Dataset()
    for dataset in mV.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, mV.experiments[0])
        current_length = da.sizes['time']
        da = xr.DataArray(da.data, dims=['time'], coords={'time': np.arange(0, current_length)})
        if current_length < max_length:
            padding = xr.DataArray(np.full((max_length - current_length,), np.nan), dims=['time'], coords={'time': np.arange(current_length, max_length)})
            da = xr.concat([da, padding], dim='time')
        ds_y[dataset] = da
    # print(ds_y)
    # exit()
    ymin, ymax = None, None
    for var_name, data_array in ds_y.data_vars.items():
        min_var = data_array.min().values
        max_var = data_array.max().values
        if ymin is None or min_var < ymin:
            ymin = min_var
        if ymax is None or max_var > ymax:
            ymax = max_var
    # print(ymin, ymax)
            

# --------------------------------------------------------------------------------------- test plot --------------------------------------------------------------------------------------------------- #
    switch = {
        'delete_previous_plots':    True,
        'basic':                    True,
        }

    mFp.remove_test_plots() if switch['delete_previous_plots'] else None
    # exit()
    
    if switch['basic']:    
        filename = 'basic.png'
        fig, axes = plot_dsScatter(ds_x, ds_y, variable_list = list(ds_x.data_vars.keys()), fig_title = 'Correlation test', label_x = '', label_y = '', colors_scatter = ['k']*len(ds_x.data_vars), colors_slope = ['k']*len(ds_x.data_vars), cmap = 'Blues',
                        xmin = None, xmax = None, ymin = None, ymax = None, 
                        density_map = False, models_highlight = ['a', 'b'],
                        fig_given = False, ax = '')
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)


