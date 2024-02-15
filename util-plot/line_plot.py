''' 
# ----------------
#   Line plot
# ----------------
This script plots line plots (like time-series plots)
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ------------------------
#    Plot functions
# ------------------------
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

def generate_distinct_colors(n):
    hsv_colors = [(i / n, 1, 1) for i in range(n)]  # Hue varies, saturation and value are maxed
    rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsv_colors]
    return rgb_colors
    

# ---------------------------------------------------------------------------------------- specific --------------------------------------------------------------------------------------------------- #
def format_fig(num_subplots):
    ncols = 4 if num_subplots > 4 else num_subplots # max 4 subplots per row
    nrows = int(np.ceil(num_subplots / ncols))
    width, height = [14, 3]     if nrows == 1   else [14, 6] 
    # width, height = [14, 3.5]   if nrows == 2   else [width, height]
    # width, height = [14, 4.5]   if nrows == 3   else [width, height]
    # width, height = [14, 5]     if nrows == 4   else [width, height]
    width, height = [12, 8.5]   if nrows == 5   else [12, 8.5] 
    width, height = [12, 10]    if nrows == 6   else [width, height]
    width, height = [12, 11.5]  if nrows == 7   else [width, height]
    width, height = [12, 11.5]  if nrows == 8   else [width, height]
    fig, axes = create_figure(width = width, height = height, nrows=nrows, ncols=ncols)
    delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, axes, nrows, ncols

def format_axes(fig, ax, subplot, num_subplots, nrows, ncols, axtitle, label_x, label_y,
                xmin = None, xmax = None, ymin = None, ymax = None, title=''):
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
    scale_ax_x(ax, 0.9)
    scale_ax_y(ax, 0.85)
    plot_axtitle(fig, ax, axtitle[subplot], xpad = 0.05, ypad = 0.0075, fontsize = 10)
    plot_xlabel(fig, ax, label_x, pad=0.055, fontsize = 10)                             if subplot >= num_subplots-ncols else None
    plot_ylabel(fig, ax, label_y, pad = 0.0475, fontsize = 10)                          if col == 0 else None
    ax.set_xticklabels([])                                                              if not subplot >= num_subplots-ncols else None
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    fig.text(0.5, 0.985, title, ha = 'center', fontsize = 9, transform=fig.transFigure)
    
def format_ax(fig, ax, label_x, label_y, xmin, xmax, ymin, ymax, title):
    plot_xlabel(fig, ax, label_x, pad=0.055, fontsize = 10)
    plot_ylabel(fig, ax, label_y, pad = 0.0475, fontsize = 10)
    ax.set_xlim([xmin, xmax])
    ax.set_ylim([ymin, ymax])
    fig.text(0.5, 0.985, title, ha = 'center', fontsize = 9, transform=fig.transFigure)


def plot_ax_line(ax, x = None, y = None, color = 'k', linewidth = None):
    if x is None:
        h = ax.plot(y, color = color, linewidth = linewidth)
    else:
        h = ax.plot(x, y, color = color, linewidth = linewidth)
    return h

def plot_dsLine(ds, x = None, variable_list = ['a', 'b'], title = '', label_x = '', label_y = '', colors = 'k', 
                xmin = None, xmax = None, ymin = None, ymax = None,
                fig_given = False, one_ax = False, fig = '', axes = ''):
    if not fig_given:  
            [fig, axes, nrows, ncols] = format_fig(len(ds.data_vars)) if not one_ax else format_fig(1)
    for subplot, variable in enumerate(ds.data_vars):
        y = ds[variable]
        ax = axes.flatten()[subplot] if not one_ax else axes
        h = plot_ax_line(ax, x, y, color = colors[subplot], linewidth = None)
        if not fig_given:  
            format_axes(fig, ax, subplot, len(ds.data_vars), nrows, ncols, variable_list, label_x, label_y, xmin, xmax, ymin, ymax, title) if not one_ax else format_ax(fig, ax, label_x, label_y, xmin, xmax, ymin, ymax, title)
        # plt.grid(True)
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
    # exit()
    ymin = None
    ymax = None
    for var_name, data_array in ds.data_vars.items():
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
        'basic':                    False,
        'one_ax':                   True,
        'fig_given':                False
        }

    mFp.remove_test_plots() if switch['delete_previous_plots'] else None
    # exit()
    
    if switch['basic']:
        filename = 'basic.png'
        fig, axes = plot_dsLine(ds, x = None, variable_list = ['a', 'b'], title = 'ROME (timeseries)', label_x = 'time [Nb]', label_y = r'ROME [km$^2$]', colors = ['k']*len(ds.data_vars), 
                    ymin = ymin, ymax = ymax,
                    fig_given = False, one_ax = False)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['one_ax']:
        filename = 'one_ax.png'
        colors = generate_distinct_colors(len(ds.data_vars))
        fig, axes = plot_dsLine(ds, x = None, variable_list = ['a', 'b'], title = 'ROME (timeseries)', label_x = 'time [Nb]', label_y = r'ROME [km$^2$]', colors = colors, 
                    ymin = ymin, ymax = ymax,
                    fig_given = False, one_ax = True)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['fig_given']:
        colors = generate_distinct_colors(len(ds.data_vars))
        filename = 'fig_given.png'
        fig, axes = plot_dsLine(ds, x = None, variable_list = ['a', 'b'], title = 'ROME (timeseries)', label_x = 'time [Nb]', label_y = r'ROME [km$^2$]', colors = ['k']*len(ds.data_vars), 
                    ymin = ymin, ymax = ymax,
                    fig_given = False, one_ax = False)
        
        for dataset in mV.datasets:
            ds[dataset] = ds[dataset] * 6
        fig, axes = plot_dsLine(ds, x = None, variable_list = ['a', 'b'], title = 'ROME (timeseries)', label_x = 'time [Nb]', label_y = r'ROME [km$^2$]', colors = colors, 
                    ymin = ymin, ymax = ymax,
                    fig_given = True, one_ax = False, fig = fig, axes = axes)
        mFp.show_plot(fig, show_type = 'save_cwd', filename = filename)






