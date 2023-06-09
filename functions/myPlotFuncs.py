import xarray as xr
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt
import seaborn as sns
import cartopy.crs as ccrs
import cartopy.feature as cfeat

import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)



# ---------------------------------------------------------------------------------------- basic plot functions ------------------------------------------------------------------------------------------------------- #

def plot_xr_scene(scene, cmap='Reds', zorder= 0, title='', ax='', vmin=None, vmax=None, fig_width=17.5, fig_height=8):
    projection = ccrs.PlateCarree(central_longitude=180)
    lat = scene.lat
    lon = scene.lon

    # if the scene is plotted as a figure by itself
    if not ax:
        f, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=(fig_width, fig_height))
        pcm = scene.plot(transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.add_feature(cfeat.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
        
        ax.set_xticks([-180, -90, 0, 90, 180])
        ax.set_yticks([-20, 0, 20])
        ax.set_xticklabels([0, 90, 180, 270, 360])
        plt.tight_layout()

    # if the scene is plotted as subplots in a larger figure
    else:
        lonm,latm = np.meshgrid(lon,lat)
        ax.add_feature(cfeat.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

        pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title)

    return pcm


def plot_scene(da, fig_width, fig_height):
    plt.figure(fig_width, fig_height)








def plot_timeseries(y, timeMean_option='', title='', ax='', ymin=None, ymax=None, fig_width=20, fig_height=5):

    if not ax:
        f, ax = plt.subplots(figsize=(fig_width, fig_height))

    if timeMean_option == 'seasonal':
        ax.plot(y, label = y.season.values)
    else:
        ax.plot(y)
        ax.axhline(y=y.mean(dim='time'), color= 'k',  linestyle="--")
    
    ax.set_title(title)
    ax.set_ylim([ymin, ymax])



def plot_bar(y, timeMean_option=[''], title='', ax='', ymin=None, ymax=None, fig_width=20, fig_height=5):

    if not ax:
        f, ax = plt.subplots(figsize=(fig_width, fig_height))

    if not timeMean_option[0]:
        y.to_series().plot.bar(ax=ax)

    if timeMean_option[0] == 'seasonal':
        y = y.resample(time='QS-DEC').mean(dim="time")
        y = to_monthly(y)
        y = y.rename({'month':'season'})
        y = y.assign_coords(season = ["MAM", "JJA", "SON", "DJF"])
        y = y.isel(year=slice(1, None))
        y= (y.mean(dim='year') - y.mean(dim='year').mean(dim='season'))

        y.to_series().plot.bar(ax=ax)
        ax.axhline(y=0, color='k',linestyle='--')
        ax.set_xticklabels(y.season.values, rotation=30, ha='right')

    if timeMean_option[0] == 'monthly':
        y = to_monthly(y)
        y = y.assign_coords(month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul','aug', 'sep', 'oct', 'nov', 'dec'])
        y= (y.mean(dim='year') - y.mean(dim='year').mean(dim='month'))

        # ax.plot(y)
        y.to_series().plot.bar(ax=ax)
        ax.axhline(y=0, color='k',linestyle='--')
        ax.set_xticks(np.arange(0,12))
        ax.set_xticklabels(y.month.values,rotation=30, ha='right')
    
    ax.set_ylim([ymin, ymax])
    ax.set_title(title)



def plot_boxplot(y, title='', ylabel='', ax=''):

    if not ax:
        plt.figure(figsize=(4,6))

    plt.xlim(0,1)
    plt.boxplot(y,vert=True, positions= [0.3], patch_artist=True, medianprops = dict(color="b",linewidth=1),boxprops = dict(color="b",facecolor='w',zorder=0)
                ,sym='+',flierprops = dict(color="r"))

    x = np.linspace(0.3-0.025, 0.3+0.025, len(y))
    plt.scatter(x, y, c='k', alpha=0.4)

    plt.xticks([])
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend(bbox_to_anchor=(0.6,0.5,0.4,0.4))
    sns.despine(top=True, right=True, left=False, bottom=True)


def plot_scatter(x,y,ax, color='k'):
    ax.scatter(x,y,facecolors='none', edgecolor=color)
    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(0.8, 0.875), textcoords='axes fraction') # xy=(0.2, 0.1), xytext=(0.05, 0.875)


def plot_bins(x,y, ax, color='k'):    
    bin_width = (x.max() - x.min())/100
    bin_end = x.max()
    bins = np.arange(0, bin_end+bin_width, bin_width)

    y_bins = []
    for i in np.arange(0,len(bins)-1):
        y_bins = np.append(y_bins, y.where((x>=bins[i]) & (x<=bins[i+1])).mean())
    ax.plot(bins[:-1], y_bins, color)

    res= stats.pearsonr(x,y)
    if res[1]<=0.05:
        ax.annotate('R$^2$: '+ str(round(res[0]**2,3)), xy=(0.2, 0.1), xycoords='axes fraction', xytext=(0.8, 0.875), textcoords='axes fraction')










# ---------------------------------------------------------------------------------------------- Formatting ----------------------------------------------------------------------------------------------------------- #

markers = ['o','d','X','s','D', '^','v', '<', '>', 'P']

def get_super(x):
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)


def orderByTas(use = True, datasets=[''], experiment = 'historical', resolution = 'regridded'):
    if use:
        order_list = []
        for dataset in datasets:
            tas = get_metric('tas_sMean', dataset, experiment=experiment, resolution=resolution)['tas_sMean'].mean(dim='time')
            order_list = np.append(order_list, tas)
        order = np.argsort(order_list)[::-1]
        n = len(datasets)
        n_half = n // 2
        colors = ['#EE0000'] * n_half + ['royalblue'] * n_half
        if n % 2 == 1:
            colors.append('royalblue')
    else:
        n = len(datasets)
        order = np.arange(n, dtype=int)
        colors = ['black'] * n
    return order, colors


def orderByTasdiff(use = True, datasets=[''], models_cmip5 = [''], resolution = 'regridded'):
    if use:
        order_list = []
        for dataset in datasets:
            tas_historical = get_metric('tas_sMean', dataset, experiment='historical', resolution=resolution)['tas_sMean'].mean(dim='time')
            if np.isin(models_cmip5, dataset).any():
                tas_rcp = get_metric('tas_sMean', dataset, experiment='rcp85', resolution=resolution)['tas_sMean'].mean(dim='time')
            else:
                tas_rcp = get_metric('tas_sMean', dataset, experiment='ssp585', resolution=resolution)['tas_sMean'].mean(dim='time')
            tasdiff = tas_rcp - tas_historical
            order_list = np.append(order_list, tasdiff)
        order = np.argsort(order_list)[::-1]
        n = len(datasets)
        n_half = n // 2
        colors = ['#EE0000'] * n_half + ['royalblue'] * n_half
        if n % 2 == 1:
            colors.append('royalblue')
    else:
        n = len(datasets)
        order = np.arange(n, dtype=int)
        colors = ['black'] * n
    return order, colors











# ---------------------------------------------------------------------------------------------- Different ways to plot ----------------------------------------------------------------------------------------------------------- #




# ------------ 
# fig = plt.figure(figsize=(fig_width, fig_height))
# fig.add_axes
# ------------

# def plot_multiple_scenes(scene, cmap, title, cbar_label='', fig_width = 20, fig_height=10, vmin=None, vmax=None, zorder=0):
#     projection = ccrs.PlateCarree(central_longitude=180)
#     lat = scene.lat
#     lon = scene.lon

#     fig = plt.figure(figsize=(fig_width, fig_height)) # in inches (letter paper: 8.5x11.0) (a4 paper: 8.3 x 11.7, right margin: 0.52, bottom: 1.44) 

#     nrows, ncols = 4, 4
#     subplot_width = 0.8 / ncols
#     subplot_height = 0.8 / nrows
#     subplot_start_x = 0.1
#     subplot_start_y = 0.9  # start at the top of the figure

#     i=0
#     scene = mF.load_metric('pr', 'rxday_snapshot', mV.datasets[i])['rx1day']
#     for row in range(nrows):
#         for col in range(ncols):
#             bottom = subplot_start_y - (row+1)*subplot_height  # decrease y-coordinate with each row
#             left = subplot_start_x + col*subplot_width
#             ax = fig.add_axes([left, bottom, subplot_width, subplot_height], projection=projection)

#             pcm = ax.pcolormesh(lon, lat, scene, transform=ccrs.PlateCarree(), zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)
#             ax.add_feature(cfeat.COASTLINE)
#             ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
#             format_mapPlot(ax, title, row, col, ncols, i)
#             i += 1
#             if i==1:
#                 return

#     return fig, pcm



# def plot_scene(scene, cmap, cbar_label, title, fig_width=12, fig_height=4, vmin=None, vmax=None, zorder=0):
#     projection = ccrs.PlateCarree(central_longitude=180)
#     lat = scene.lat
#     lon = scene.lon

#     fig = plt.figure(figsize=(fig_width, fig_height))
#     ax = fig.add_axes([0.08, 0.15, 0.85, 1], projection=projection) # Specify position [left, bottom, width, height]
#     pcm = ax.pcolormesh(lon, lat, scene, transform=ccrs.PlateCarree(), zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)
#     ax.add_feature(cfeat.COASTLINE)
#     ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

#     xticks = [-180, -90, 0, 90, 180]
#     yticks = [-20, 0, 20]
#     ax.set_xticks(xticks)
#     ax.set_yticks(yticks)

#     ax.text(-179, 33, f'{title}', fontsize=16)
#     ax.text(-202.5, -2, 'lat', rotation='vertical', fontsize=13)
#     ax.text(-4.5, -47, 'lon', fontsize=13)

#     cbar_ax = fig.add_axes([0.105, 0.21, 0.8, 0.04])
#     cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
#     cbar.set_label(f'{cbar_label} [{scene.units.data}]', labelpad=10) 
#     return fig, pcm, ax
# # fig, pcm, ax = plot_scene(scene, cmap='Blues', cbar_label = 'precipitation', title=title)








# ------------ 
# xr.plot()
# ------------

# a = scene.plot(transform=ccrs.PlateCarree(), cmap=cmap) # cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, vmin=vmin, vmax=vmax
# ax.add_feature(cfeat.COASTLINE)
# ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
# ax.set_title(title)
# cbar = plt.colorbar(a, ax=ax, orientation='horizontal')


# ax.collections[0].colorbar.ax.set_anchor((0., -0.2))
# ax.set_xticks([30, 90, 150, 210, 270, 330], crs=ccrs.PlateCarree())
# ax.set_yticks([-20, -10, 0, 10, 20], crs=ccrs.PlateCarree())
# ax.yaxis.set_ticks_position('both')
# ax.xaxis.set_major_formatter(LongitudeFormatter())
# ax.yaxis.set_major_formatter(LatitudeFormatter())






# ------------ 
# fig, axes = plt.subplots(nrows, ncols)
# for ax in axes.flatten():
# (or ax = axes[row, col])
# ------------

# Create the figure and axes
# fig_size= (13,6.5)
# nrows = 4
# ncols = 4
# projection = ccrs.PlateCarree(central_longitude=180)
# fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size),
#                     subplot_kw=dict(projection=projection))

# lat = scene.lat
# lon = scene.lon
# lonm,latm = np.meshgrid(lon,lat)

# # Loop through the axes
# for ax in axes.flatten():
#     ax.add_feature(cfeat.COASTLINE)
#     ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
#     pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)

#     ax_position = ax.get_position()
#     ax_left = ax_position.x0 - 0.075
#     ax_bottom = ax_position.y0 -0.075
#     ax_width = ax_position.width
#     ax_height = ax_position.height
#     ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


# Adjust the positions of the subplots
# plt.subplots_adjust(left=left, right=1-left, bottom=bottom, top=1-bottom, wspace=wspace, hspace=hspace)


# left = 0.025
# bottom = 0.075
# plt.subplots_adjust(left=left, right=1-left, bottom=bottom, top=1-bottom)


# wspace = 0.05
# hspace = - 0
# plt.subplots_adjust(wspace=wspace, hspace=hspace)


# left = 0.025
# bottom = 0.075
# plt.subplots_adjust(left=left, right=1-left, bottom=bottom, top=1-bottom)



# Show the figure
# plt.show()




# ------------ 
# fig= plt.figure(figsize=fig_size)
# ax= fig.add_subplot(nrows,ncols,i+1,projection=projection)
# ------------


# # Create the figure and axes
# fig_size= (13.5,6.5)
# nrows = 4
# ncols = 4
# projection = ccrs.PlateCarree(central_longitude=180)
# fig= plt.figure(figsize=fig_size)

# lat = scene.lat
# lon = scene.lon
# lonm,latm = np.meshgrid(lon,lat)

# # Loop through the axes
# for i in np.arange(16):
#     ax= fig.add_subplot(nrows,ncols,i+1,projection=projection)
#     ax.add_feature(cfeat.COASTLINE)
#     ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
#     pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)



    # ax_position = ax.get_position()
    # ax_left = ax_position.x0
    # ax_bottom = ax_position.y0
    # ax_width = ax_position.width *2
    # ax_height = ax_position.height * 2
    # ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

    # if i == 3:
    #     ax_position = ax.get_position()
    #     ax_left = ax_position.x0 - 0.1 + 0.1
    #     ax_bottom = ax_position.y0
    #     ax_width = ax_position.width *1.7
    #     ax_height = ax_position.height * 1.7
    #     ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

    # else:
    #     ax_position = ax.get_position()
    #     ax_left = ax_position.x0 - 0.1
    #     ax_bottom = ax_position.y0
    #     ax_width = ax_position.width *1.7
    #     ax_height = ax_position.height * 1.7
    #     ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


# Adjust the positions of the subplots
# plt.subplots_adjust(left=left, right=1-left, bottom=bottom, top=1-bottom, wspace=wspace, hspace=hspace)


# left = 0.025
# bottom = 0.075
# plt.subplots_adjust(left=left, right=1-left, bottom=bottom, top=1-bottom)


# wspace = 0.1
# plt.subplots_adjust(wspace=wspace)


# left = 0.025
# bottom = 0.075
# plt.subplots_adjust(left=left, right=1-left, bottom=bottom, top=1-bottom)







# ------------ 
# gridspec
# ------------

# fig_size = (13.5, 6.5)
# nrows = 4
# ncols = 4

# # Create the figure and gridspec layout
# fig = plt.figure(figsize=fig_size)
# gs = fig.add_gridspec(nrows, ncols, wspace=0.1, hspace=0.1)

# lat = scene.lat
# lon = scene.lon
# lonm, latm = np.meshgrid(lon, lat)

# # Loop through the gridspec layout
# for i in range(nrows * ncols):
#     # Determine the row and column indices
#     row = i // ncols
#     col = i % ncols

#     # Create the subplot with projection
#     ax = fig.add_subplot(gs[row, col], projection=ccrs.PlateCarree())

#     # Plot features and set extent
#     ax.add_feature(cfeat.COASTLINE)
#     ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())

#     # Add pcolormesh plot
#     pcm = ax.pcolormesh(lonm, latm, scene, transform=ccrs.PlateCarree(), cmap=cmap, vmin=vmin, vmax=vmax)

# # Adjust spacing between subplots
# # fig.tight_layout()
# fig.subplots_adjust(left=0.05, right=0.95, wspace=0.1, hspace=-0.8)

# # Show the figure
# plt.show()



# import matplotlib.gridspec as gridspec

# # Define the figure size and grid dimensions
# fig_size = (8, 6)
# nrows = 3
# ncols = 3

# # Define the width ratios and height ratios
# width_ratios = [1, 2, 1]
# height_ratios = [1, 1, 2]

# # Create the figure and gridspec layout
# fig = plt.figure(figsize=fig_size)
# gs = fig.add_gridspec(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)

# # Create subplots within the gridspec layout
# for i in range(nrows):
#     for j in range(ncols):
#         ax = fig.add_subplot(gs[i, j], frameon=True)
#         ax.plot([0, 1], [0, 1], 'k-')
#         ax.set_xlim(0, 1)
#         ax.set_ylim(0, 1)

# # Show the figure
# plt.show()





# ------------ 
# other
# ------------


# new_bottom = 0.425 # When picking out the tropics, there is abit of black space left from the full map axis
# left, _, width, height = ax_position.bounds
# ax.set_position([left, new_bottom, width, height])





# fig = plt.figure(figsize=(fig_size))
# ax= fig.add_subplot(nrows,ncols,i+1,projection=projection)



# def move_col(ax, moveby):
#     ax_position = ax.get_position()
#     _, bottom, width, height = ax_position.bounds
#     new_left = _ + moveby
#     ax.set_position([new_left, bottom, width, height])
#     return

# def move_row(ax, moveby):
#     ax_position = ax.get_position()
#     left, _, width, height = ax_position.bounds
#     new_bottom = _ + moveby
#     ax.set_position([left, new_bottom, width, height])

# def scale_ax(ax, scaleby):
#     ax_position = ax.get_position()
#     left, bottom, _1, _2 = ax_position.bounds
#     new_width = _1 * scaleby
#     new_height = _2 * scaleby
#     ax.set_position([left, bottom, new_width, new_height])


# def plot_multiple_scenes_rel(variable, metric, metric_option, cmap, title = '', datasets = mV.datasets, cbar_label = ''):
#     # create figure
#     nrows = 4
#     ncols = 4
#     fig_size= (14,5.5)
#     dataset = datasets

#     projection = ccrs.PlateCarree(central_longitude=180)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size), subplot_kw=dict(projection=projection))

#     for i in np.arange(16): #, dataset in enumerate(datasets):
#         scene = mF.load_metric(variable, metric, dataset)[metric_option]

#         # Determine the row and column indices
#         row = i // ncols
#         col = i % ncols

#         # plot
#         ax = axes.flatten()[i]
#         lat = scene.lat
#         lon = scene.lon
#         lonm,latm = np.meshgrid(lon,lat)
#         ax.add_feature(cfeat.COASTLINE)
#         ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
#         pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)

#         # Adjust axis position and size
#         if col == 0:
            # ax_position = ax.get_position()
            # ax_left = ax_position.x0 -0.0825
            # ax_bottom = ax_position.y0
            # ax_width = ax_position.width *1.3
            # ax_height = ax_position.height *1.3
            # ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

#         if col == 1:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0 -0.0435
#             ax_bottom = ax_position.y0
#             ax_width = ax_position.width *1.3
#             ax_height = ax_position.height *1.3
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


#         if col == 2:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0 - 0.005
#             ax_bottom = ax_position.y0
#             ax_width = ax_position.width *1.3
#             ax_height = ax_position.height *1.3
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


#         if col == 3:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0 + 0.0325
#             ax_bottom = ax_position.y0
#             ax_width = ax_position.width *1.3
#             ax_height = ax_position.height *1.3
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


#         if row == 0:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0
#             ax_bottom = ax_position.y0 + 0.04
#             ax_width = ax_position.width
#             ax_height = ax_position.height
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

#         if row == 1:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0
#             ax_bottom = ax_position.y0 + 0.04
#             ax_width = ax_position.width
#             ax_height = ax_position.height
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


#         if row == 2:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0
#             ax_bottom = ax_position.y0 + 0.04
#             ax_width = ax_position.width
#             ax_height = ax_position.height
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


#         if row == 3:
#             ax_position = ax.get_position()
#             ax_left = ax_position.x0
#             ax_bottom = ax_position.y0 + 0.04
#             ax_width = ax_position.width
#             ax_height = ax_position.height
#             ax.set_position([ax_left, ax_bottom, ax_width, ax_height])


#         # Create colorbar axis and plot colorbar
#         if row == 3:
#             ax_position = ax.get_position()
#             colorbar_height = 0.01
#             colorbar_bottom = ax_position.y0 - colorbar_height -0.09  # Adjust the padding of the colobar
#             colorbar_left = ax_position.x0
#             colorbar_width = ax_position.width
#             cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
#             cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
#             cbar.ax.tick_params(labelsize=8)

#         else:
#             ax_position = ax.get_position()
#             colorbar_height = 0.01
#             colorbar_bottom = ax_position.y0 - colorbar_height -0.0175  # Adjust the padding of the colobar
#             colorbar_left = ax_position.x0
#             colorbar_width = ax_position.width
#             cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
#             cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
#             cbar.ax.tick_params(labelsize=8)


#         # Plot text
#         # lon
#         if row == nrows-1:
#             ax_position = ax.get_position()
#             lon_text_x =  ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2 #0
#             lon_text_y =  colorbar_bottom + colorbar_height + 0.02 # -45
#             ax.text(lon_text_x, lon_text_y, 'Lon', ha = 'center', fontsize = 8, transform=fig.transFigure)

#         # lat
#         if col == 0:
#             ax_position = ax.get_position()
#             ax_position = ax.get_position()
#             lat_text_x = ax_position.x0 -0.0375 #-200
#             lat_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2 #0
#             ax.text(lat_text_x, lat_text_y, 'Lat', va = 'center', rotation='vertical', fontsize = 8, transform=fig.transFigure)

#         # colobar label
#         if row == nrows-1:    
#             ax_position = ax.get_position()
#             cbar_text_x = ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
#             cbar_text_y = colorbar_bottom -0.065
#             ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 9, transform=fig.transFigure)

#         # titles
#         ax_title = 'dataset'
#         title_text_x = -175
#         title_text_y = 33
#         ax.text(title_text_x, title_text_y, ax_title) #, fontsize = 15)

#         title_text_x = 0.5
#         title_text_y = 0.95
#         ax.text(title_text_x, title_text_y, title, ha = 'center', fontsize = 15, transform=fig.transFigure)

#         # format ticks
#         if row == nrows-1:
#             xticks = [30, 90, 150, 210, 270, 330]
#             ax.set_xticks(xticks, crs=ccrs.PlateCarree())
#             ax.xaxis.set_major_formatter(LongitudeFormatter())
#             ax.xaxis.set_tick_params(labelsize=8)

#         if col == 0:
#             yticks = [-20, 0, 20]
#             ax.set_yticks(yticks, crs=ccrs.PlateCarree())
#             ax.yaxis.set_major_formatter(LatitudeFormatter())
#             ax.yaxis.set_tick_params(labelsize=8)
#             ax.yaxis.set_ticks_position('both')












# plotting multiple scenes with individual colorbars (relative limits)
# plot_multiple_scenes_relativeLim(variable, metric, metric_option, cmap, title = title, datasets = mV.datasets[0], cbar_label = cbar_label)
# plt.show()

# def create_cbar(fig, ax, pcm, cbar_height, pad):
#     ax_position = ax.get_position()
#     cbar_bottom = ax_position.y0 - cbar_height - pad
#     cbar_left = ax_position.x0
#     cbar_width = ax_position.width
#     cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
#     cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')
#     cbar.ax.tick_params(labelsize=8)
#     return cbar_left, cbar_bottom, cbar_width, cbar_height

# def plot_multiple_scenes_relativeLim(variable, metric, metric_option, cmap, title = '', datasets = mV.datasets, cbar_label = ''):
#     # create figure
#     nrows = 4
#     ncols = 4
#     fig_size= (14,5.5)
#     dataset = datasets

#     projection = ccrs.PlateCarree(central_longitude=180)
#     fig, axes = plt.subplots(nrows, ncols, figsize=(fig_size), subplot_kw=dict(projection=projection))

#     for i in np.arange(16): #, dataset in enumerate(datasets):
#         scene = mF.load_metric(variable, metric, dataset)[metric_option]

#         # Determine the row and column indices
#         row = i // ncols
#         col = i % ncols

#         # plot
#         ax = axes.flatten()[i]
#         lat = scene.lat
#         lon = scene.lon
#         lonm,latm = np.meshgrid(lon,lat)
#         ax.add_feature(cfeat.COASTLINE)
#         ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
#         pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)

#         # Adjust axis position and size (columns and rows at a time, then adjust scale)
#         scale_ax(ax, scaleby=1.3)

#         if col == 0:
#             move_col(ax, moveby = -0.0825)
#         if col == 1:
#             move_col(ax, moveby = -0.0435)
#         if col == 2:
#             move_col(ax, moveby = - 0.005)
#         if col == 3:
#             move_col(ax, moveby = 0.0325)


#         if row == 0:
#             move_row(ax, moveby = + 0.04)
#         if row == 1:
#             move_row(ax, moveby = + 0.04)
#         if row == 2:
#             move_row(ax, moveby = + 0.04)
#         if row == 3:
#             move_row(ax, moveby = + 0.04)


#         # Create colorbar axis and plot colorbar
#         if row == 3:
#             _, cbar_bottom, _, _ = create_cbar(fig, ax, pcm, cbar_height = 0.01, pad = 0.09)
#         else:
#             _, cbar_bottom, _, _ = create_cbar(fig, ax, pcm, cbar_height = 0.01, pad = 0.0175)

#         # Plot text
#         # lon
#         if i >= len(np.arange(14))-nrows:
#             ax_position = ax.get_position()
#             lon_text_x =  ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2 #0
#             lon_text_y =  ax_position.y0 - 0.07 # -45
#             ax.text(lon_text_x, lon_text_y, 'Lon', ha = 'center', fontsize = 8, transform=fig.transFigure)

#         # lat
#         if col == 0:
#             ax_position = ax.get_position()
#             ax_position = ax.get_position()
#             lat_text_x = ax_position.x0 -0.0375 #-200
#             lat_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2 #0
#             ax.text(lat_text_x, lat_text_y, 'Lat', va = 'center', rotation='vertical', fontsize = 8, transform=fig.transFigure)

#         # colobar label
#         if row == nrows-1:    
#             ax_position = ax.get_position()
#             cbar_text_x = ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
#             cbar_text_y = cbar_bottom -0.065
#             ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 9, transform=fig.transFigure)

#         # titles
#         ax_position = ax.get_position()
#         ax_title = 'dataset'
#         title_text_x = ax_position.x0 + 0.002
#         title_text_y = ax_position.y1 + 0.0095
#         ax.text(title_text_x, title_text_y, ax_title, fontsize = 9, transform=fig.transFigure)

#         title_text_x = 0.5
#         title_text_y = 0.95
#         ax.text(title_text_x, title_text_y, title, ha = 'center', fontsize = 15, transform=fig.transFigure)

#         format_ticks(ax, i, nrows, col, fontsize=9)
#     return fig




# def plot_scene(scene, cmap, title = '', vmin = None, vmax = None, cbar_label = '', fig_size=(12,4), zorder=0):
#     # create figure
#     projection = ccrs.PlateCarree(central_longitude=180)
#     fig, ax = plt.subplots(figsize=(fig_size), subplot_kw=dict(projection=projection))

#     # plot
#     lat = scene.lat
#     lon = scene.lon
#     lonm,latm = np.meshgrid(lon,lat)
#     ax.add_feature(cfeat.COASTLINE)
#     ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())    
#     pcm = ax.pcolormesh(lonm,latm, scene, transform=ccrs.PlateCarree(),zorder=zorder, cmap=cmap, vmin=vmin, vmax=vmax)

#     # Adjust axis position
#     ax_position = ax.get_position()
#     ax_left = ax_position.x0 -0.055
#     ax_bottom = ax_position.y0 + 0.075
#     ax_width = ax_position.width *1.15
#     ax_height = ax_position.height *1.15
#     ax.set_position([ax_left, ax_bottom, ax_width, ax_height])

#     # Create colorbar axis and plot colorbar
#     ax_position = ax.get_position()
#     colorbar_height = 0.05
#     colorbar_bottom = ax_position.y0 - colorbar_height - 0.15  # Adjust the padding of the colobar
#     colorbar_left = ax_position.x0
#     colorbar_width = ax_position.width
#     cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
#     cbar = fig.colorbar(pcm, cax=cbar_ax, orientation='horizontal')

#     # Plot text
#     # lon
#     lon_text_x =  ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2 #0
#     lon_text_y =  ax_position.y0 - 0.1 # -45
#     ax.text(lon_text_x, lon_text_y, 'Lon', ha = 'center', fontsize = 12, transform=fig.transFigure)

#     # lat
#     lat_text_x = ax_position.x0 -0.055 #-200
#     lat_text_y = ax_position.y0 + (ax_position.y1 - ax_position.y0) / 2 #0
#     ax.text(lat_text_x, lat_text_y, 'Lat', va = 'center', rotation='vertical', fontsize = 12, transform=fig.transFigure)

#     # colobar label
#     cbar_text_x = ax_position.x0 + (ax_position.x1 - ax_position.x0) / 2
#     cbar_text_y = colorbar_bottom -0.125
#     ax.text(cbar_text_x, cbar_text_y, cbar_label, ha = 'center', fontsize = 12, transform=fig.transFigure)

#     # title
#     title_text_x = ax_position.x0 + 0.005
#     title_text_y = ax_position.y1 + 0.025
#     ax.text(title_text_x, title_text_y, title, fontsize = 15, transform=fig.transFigure)

#     # Format ticks
#     xticks = [30, 90, 150, 210, 270, 330]
#     ax.set_xticks(xticks, crs=ccrs.PlateCarree())
#     ax.xaxis.set_major_formatter(LongitudeFormatter())

#     yticks = [-20, 0, 20]
#     ax.set_yticks(yticks, crs=ccrs.PlateCarree())
#     ax.yaxis.set_ticks_position('both')
#     ax.yaxis.set_major_formatter(LatitudeFormatter())
#     return fig





















































