import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))



ds = xr.open_dataset('/scratch/b/b382628/OutfileName.nc', chunks="auto", engine="netcdf4")
print(ds)
da = ds['pr'].sel(lat = slice(-30, 30))*60*60*24
print(format_bytes(da.nbytes))


plt.figure(figsize=(10, 5))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()  # Add coastlines
da.plot(ax=ax, transform=ccrs.PlateCarree(), vmin=0, vmax=20)
ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
plt.savefig("my_plot.png", bbox_inches='tight', dpi=300)
plt.close()


























