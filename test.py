import numpy as np
import xarray as xr
import cartopy
import matplotlib.pyplot as plt





def plot_scenes(ds, cmap='Reds', title='', vmin = None, vmax=None):
    fig= plt.figure(figsize=(22,12))
    fig.suptitle(title, fontsize=18, y=0.89)

    lat = ds.lat
    lon = ds.lon
    lonm,latm = np.meshgrid(lon,lat)

    for i, model in enumerate(models):
        ax= fig.add_subplot(5,4,i+1, projection=cartopy.crs.PlateCarree(central_longitude=180))
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=cartopy.crs.PlateCarree())

        pcm= ax.pcolormesh(lonm,latm, ds[model],transform=cartopy.crs.PlateCarree(),zorder=0, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(pcm, ax=ax, orientation='horizontal', pad=0.05, aspect=50, fraction=0.055)







