import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeat








def plot_snapshot(var, cmap, variable_name, model):
    projection = ccrs.PlateCarree(central_longitude=180)
    lat = var.lat
    lon = var.lon

    f, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=(15, 5))

    var.plot(transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, cmap=cmap)
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    ax.set_title(variable_name + ' snapshot, model:' + model)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([0, 90, 180, 270, 360])
    ax.set_yticks([-20, 0, 20])
    plt.tight_layout()






# plot binary scene
# O = pr_day.where(pr_day>=conv_threshold,0)>0
# xr_L = xr.DataArray(L, coords={'lat': lat,'lon': lon})

# projection = ccrs.PlateCarree(central_longitude=180)
# f, (ax1, ax2) = plt.subplots(nrows = 2, subplot_kw=dict(projection=projection), figsize=(15, 7))

# # objects binary
# O.plot(ax=ax1, transform=ccrs.PlateCarree(), levels =4, colors = ['w','c','k'], add_colorbar=False) 
# ax1.add_feature(cfeat.COASTLINE)
# ax1.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
# ax1.set_title('') #Snapshot of objects, model:' + model + ' exp:' + experiment)
# #ax1.set_xticks([-180, -90, 0, 90, 180])
# #ax1.set_xticklabels([0, 90, 180, 270, 360])
# ax1.set_yticks([-20, 0, 20])
# plt.tight_layout()
    

# pr_day.plot(ax=ax2, transform=ccrs.PlateCarree(), levels =len(np.unique(L)),cmap='Blues',cbar_kwargs={'orientation': 'horizontal','pad':0.175, 'aspect':55,'fraction':0.055})#,add_colorbar=False)
# ax2.add_feature(cfeat.COASTLINE)
# ax2.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
# ax2.set_title('') #Snapshot of objects, model:' + model + ' exp:' + experiment)
# ax2.set_xticks([-180, -90, 0, 90, 180])
# ax2.set_xticklabels([0, 90, 180, 270, 360])
# ax2.set_yticks([-20, 0, 20])

# plt.tight_layout()





# # plot comparison plot
# projection = ccrs.PlateCarree(central_longitude=180)

# f, (ax1, ax2) = plt.subplots(nrows = 2, subplot_kw=dict(projection=projection), figsize=(15, 9))

# # Rx1day
# rx1day_tMean.plot(ax=ax1, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055},cmap='Blues')
# ax1.add_feature(cfeat.COASTLINE)
# ax1.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
# ax1.set_title('Rx1day_tMean, model:' + model + ' exp:' + experiment)
# ax1.set_xticks([-180, -90, 0, 90, 180])
# ax1.set_xticklabels([0, 90, 180, 270, 360])
# ax1.set_yticks([-20, 0, 20])

# # Rx5day
# rx5day_tMean.plot(ax=ax2, transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055},cmap='Blues')
# ax2.add_feature(cfeat.COASTLINE)
# ax2.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
# ax2.set_title('Rx5day_tMean, model:' + model + ' exp:' + experiment)
# ax2.set_xticks([-180, -90, 0, 90, 180])
# ax2.set_xticklabels([0, 90, 180, 270, 360])
# ax2.set_yticks([-20, 0, 20])

# plt.tight_layout()
