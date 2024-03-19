''' 
# ------------------------
#       Map plot
# ------------------------
This script plot scenes with coastlines in the background

To use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-plot')
import map_plot as mP

filename = 'map_plot.png'
fig, ax = mP.plot_dsScenes(ds, label = label, title = '', vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
'''



# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.colors as mcolors
import os
home = os.path.expanduser("~")



# ------------------------
#    Plot functions
# ------------------------
# ------------------------------------------------------------------------------------ General (from myFuncs_plots) --------------------------------------------------------------------------------------------------- #
def create_map_figure(width = 12, height = 4, nrows = 1, ncols = 1, projection = ccrs.PlateCarree(central_longitude=180)):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width,height), subplot_kw=dict(projection=projection))
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

def scale_ax(ax, scaleby):
    ax_position = ax.get_position()
    left, bottom, _1, _2 = ax_position.bounds
    new_width = _1 * scaleby
    new_height = _2 * scaleby
    ax.set_position([left, bottom, new_width, new_height])

def plot_axtitle(fig, ax, title, xpad, ypad, fontsize):
    ax_position = ax.get_position()
    title_text_x = ax_position.x0 + xpad 
    title_text_y = ax_position.y1 + ypad
    ax.text(title_text_x, title_text_y, title, fontsize = fontsize, transform=fig.transFigure)

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

def plot_colobar(fig, ax, pcm, label, variable_list = ['a', 'b']):
    if len(variable_list) > 4:
        cbar_position = [0.225, 0.095, 0.60, 0.02]      # [left, bottom, width, height]
    else:
        cbar_position = [0.225, 0.15, 0.60, 0.05] 
    cbar_ax = fig.add_axes(cbar_position)
    fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    ax.text(cbar_position[0] + cbar_position[2] / 2 , cbar_position[1]-0.075, label, ha = 'center', fontsize = 10, transform=fig.transFigure)

def cbar_below_axis(fig, ax, pcm, cbar_height, pad, numbersize = 8, cbar_label = '', text_pad = 0.1):
    # colorbar position
    ax_position = ax.get_position()
    cbar_bottom = ax_position.y0 - cbar_height - pad
    cbar_left = ax_position.x0
    cbar_width = ax_position.width
    cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
    cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
    cbar.ax.tick_params(labelsize=numbersize)
    # colobar label
    cbar_text_x = ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
    cbar_text_y = cbar_bottom - text_pad
    ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 12, transform=fig.transFigure)
    return cbar

def format_ticks(ax, i = 0, num_subplots = 1, ncols = 1, col = 0, labelsize = 8, xticks = [30, 90, 150, 210, 270, 330], yticks = [-20, 0, 20]):
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_xticklabels('')
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_yticklabels('')
    if i >= num_subplots-ncols:
        ax.xaxis.set_major_formatter(LongitudeFormatter())
        ax.xaxis.set_tick_params(labelsize=labelsize)
    if col == 0:
        ax.yaxis.set_major_formatter(LatitudeFormatter())
        ax.yaxis.set_tick_params(labelsize=labelsize)
        ax.yaxis.set_ticks_position('both')

def plot_scene(scene, cmap = 'Blues', label = '[units]', fig_title = 'test', ax_title= 'test', vmin = None, vmax = None, cat_cmap = False):
    fig, ax = create_map_figure(width = 12, height = 4)
    if cat_cmap:
        cmap = get_cat_colormap()
    pcm = plot_axMapScene(ax, scene, cmap, vmin = vmin, vmax = vmax)
    move_col(ax, moveby = -0.055)
    move_row(ax, moveby = 0.075)
    scale_ax(ax, scaleby = 1.15)
    cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = label, text_pad = 0.125)
    plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
    plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
    ax.text(0.5, 0.925, fig_title, ha = 'center', fontsize = 15, transform=fig.transFigure)
    plot_axtitle(fig, ax, ax_title, xpad = 0.005, ypad = 0.025, fontsize = 15)
    format_ticks(ax, labelsize = 11)
    return fig, ax

def save_figure(figure, folder =f'{home}/Documents/phd', filename = 'test.pdf', path = ''):
    ''' Basic plot saving function '''
    if folder and filename:
        path = os.path.join(folder, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.remove(path) if os.path.exists(path) else None
    figure.savefig(path)

def show_plot(fig, show_type = 'cycle', cycle_time = 0.5, filename = 'test'):
    ''' If using this on supercomputer, x11 forwarding is required with XQuartz installed on your computer '''
    if show_type == 'cycle':
        plt.ion()
        plt.show()
        plt.pause(cycle_time)
        plt.close(fig)
        plt.ioff()
    elif show_type == 'save_cwd':
        save_figure(figure = fig, folder = f'{os.getcwd()}/zome_plots', filename = filename)
        plt.close(fig)
        print(f'saved {filename}')
        return True
    elif show_type == 'show':
        plt.show()
        return True
    
def remove_test_plots(directory = f'{os.getcwd()}/zome_plots'):
    if os.path.exists(directory) and os.path.isdir(directory):
        png_files = [f for f in os.listdir(directory) if f.endswith('.png') or f.endswith('.pdf')]
        if png_files:
            print(f'From folder: {directory}')
            for filename in png_files:
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"File {file_path} has been removed")
        else:
            print(f"No .png or .pdf files found in {directory}")
    else:
        print(f"Directory {directory} does not exist or is not a directory")
        
def blend_with_white(color, blend_factor):
    # Convert hex to RGB if necessary
    if isinstance(color, str):
        color = mcolors.hex2color(color)
    white = np.array([1, 1, 1])
    blended_color = (1 - blend_factor) * np.array(color) + blend_factor * white
    return blended_color

def get_cat_colormap():
    original_colors = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', 
                    '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', 
                    '#000000'] + ["#"+''.join([np.random.choice(list('0123456789ABCDEF')) for j in range(6)]) for i in range(30-22)]
    blend_factor = 0.6  # Adjust this to make colors more or less dim
    dim_colors = ['#ffffff']  # Start with white for the background
    dim_colors += [blend_with_white(color, blend_factor) for color in original_colors]
    cmap = mcolors.ListedColormap(dim_colors, name='custom_dim')
    return cmap

# ---------------------------------------------------------------------------------------- specific --------------------------------------------------------------------------------------------------- #
def format_fig(num_subplots):
    ncols = 4 if num_subplots > 4 else num_subplots # max 4 subplots per row
    nrows = int(np.ceil(num_subplots / ncols))
    width, height = [14, 3]     if nrows == 1   else [14, 6] 
    width, height = [14, 3.5]   if nrows == 2   else [width, height]
    width, height = [14, 4.5]   if nrows == 3   else [width, height]
    width, height = [14, 5]     if nrows == 4   else [width, height]
    width, height = [14, 6]     if nrows == 5   else [width, height] 
    width, height = [14, 7.5]   if nrows == 6   else [width, height]
    width, height = [14, 8.25]  if nrows == 7   else [width, height]
    width, height = [14, 8.25]  if nrows == 8   else [width, height]
    fig, axes = create_map_figure(width = width, height = height, nrows=nrows, ncols=ncols)
    delete_remaining_axes(fig, axes, num_subplots, nrows, ncols)
    return fig, axes, nrows, ncols

def format_axes(fig, ax, subplot, num_subplots, nrows, ncols, axtitle):
    row, col = subplot // ncols, subplot % ncols    # determine row and col index
    move_col(ax, -0.0825 + 0.0025) if col == 0 else None
    move_col(ax, -0.0435 + 0.0025) if col == 1 else None
    move_col(ax, -0.005 + 0.0025)  if col == 2 else None
    move_col(ax, 0.0325 + 0.0025)  if col == 3 else None
    move_row(ax, 0.025+0.005)      if row == 0 else None
    move_row(ax, 0.04+0.005)       if row == 1 else None
    move_row(ax, 0.045+0.01)       if row == 2 else None
    move_row(ax, 0.05+0.01)        if row == 3 else None
    move_row(ax, 0.05+0.01)        if row == 4 else None
    move_row(ax, 0.05+0.01)        if row == 5 else None
    move_row(ax, 0.05+0.01)        if row == 6 else None
    move_row(ax, 0.05+0.01)        if row == 7 else None
    scale_ax(ax, 1.3)
    plot_xlabel(fig, ax, 'Lon', pad = 0.064, fontsize = 8) if subplot >= num_subplots-ncols else None
    plot_ylabel(fig, ax, 'Lat', pad = 0.0375, fontsize = 8) if col == 0 else None
    format_ticks(ax, subplot, num_subplots, ncols, col, labelsize = 9)
    plot_axtitle(fig, ax, axtitle, xpad = 0.002, ypad = 0.0095, fontsize = 9) if nrows < 6 else plot_axtitle(fig, ax, axtitle, xpad = 0.002, ypad = 0.0095/2, fontsize = 9) 

def plot_axMapScene(ax, scene, cmap, vmin = None, vmax = None, zorder = 0, cat_cmap = False):
    lat = scene.lat
    lon = scene.lon
    lonm,latm = np.meshgrid(lon,lat)
    ax.add_feature(cfeat.COASTLINE, edgecolor = 'limegreen')
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
    if cat_cmap:
        cmap = get_cat_colormap()
    pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)
    return pcm

def plot_dsScenes(ds, label = 'units []', title = '', vmin = None, vmax = None, cmap = 'Blues', variable_list = ['a', 'b'],
                  cat_cmap = False):
    ''' Plotting multiple scenes based on daatset with scenes '''
    if len(variable_list) == 1:
        for dataset in variable_list:
            return plot_scene(ds[dataset], cmap = cmap, label = label, fig_title = title, ax_title= dataset, vmin = vmin, vmax = vmax, cat_cmap = cat_cmap)
    fig, axes, nrows, ncols = format_fig(len(variable_list))
    for subplot, dataset in enumerate(variable_list):
        da = ds[dataset]
        top_text = 0.95 if len(variable_list) > 4 else 0.85
        ax = axes.flatten()[subplot]
        pcm = plot_axMapScene(ax, da, cmap, vmin = vmin, vmax = vmax, cat_cmap = cat_cmap)
        format_axes(fig, ax, subplot, len(variable_list), nrows, ncols, axtitle = dataset)
    fig.text(0.5, top_text, title, ha = 'center', fontsize = 15, transform=fig.transFigure)  # fig title
    plot_colobar(fig, ax, pcm, label, variable_list)
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
    import choose_datasets as cD           
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import metric_data as mD


# --------------------------------------------------------------------------------------- get data --------------------------------------------------------------------------------------------------- #
    metric_type = 'conv_org'
    metric_name = f'obj_snapshot_{cD.conv_percentiles[0]}thprctile'
    ds = xr.Dataset()
    for dataset in cD.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, cD.experiments[0])
        ds[dataset] = da>0
    # print(ds)
        

# --------------------------------------------------------------------------------------- test plot --------------------------------------------------------------------------------------------------- #
    switch = {
        'delete_previous_plots':    True,
        'separate_subplots':        False,
        'one_plot':                 True,
        }

    remove_test_plots() if switch['delete_previous_plots'] else None
    # exit()

    if switch['separate_subplots']:
        filename = 'separate_subplots.png'
        fig, ax = plot_dsScenes(ds, label = 'units []', title = '', vmin = None, vmax = None, cmap = 'Blues', variable_list = list(ds.data_vars.keys()))
        show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch['one_plot']:
        filename = 'one_plot.pdf'
        first_variable = list(ds.data_vars)[0]
        ds_one = xr.Dataset(data_vars = {first_variable : ds[first_variable]})
        fig, ax = plot_dsScenes(ds_one, label = 'units []', title = '', vmin = None, vmax = None, cmap = 'Blues', variable_list = list(ds_one.data_vars.keys()))
        show_plot(fig, show_type = 'save_cwd', filename = filename)






