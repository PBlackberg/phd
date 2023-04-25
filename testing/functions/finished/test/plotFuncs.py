def plot_snapshot(var, variable_name, cmap):
    projection = ccrs.PlateCarree(central_longitude=180)
    lat = var.lat
    lon = var.lon

    f, ax = plt.subplots(subplot_kw=dict(projection=projection), figsize=(20, 10))
    var.plot(transform=ccrs.PlateCarree(), cbar_kwargs={'orientation': 'horizontal','pad':0.125, 'aspect':50,'fraction':0.055}, cmap=cmap)
    ax.add_feature(cfeat.COASTLINE)
    ax.set_extent([lon[0], lon[-1], lat[0], lat[-1]], crs=ccrs.PlateCarree())
    ax.set_title(variable_name + ' snapshot, ' + model + ', ' + experiment)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_xticklabels([0, 90, 180, 270, 360])
    ax.set_yticks([-20, 0, 20])
    plt.tight_layout()

    
    
    
    
    
    
def plot_timeseries(y, variable_name, series_type):
    plt.figure(figsize=(25,5))
    plt.plot(y)
    plt.axhline(y=y.mean(dim='time'), color='k')
    plt.title(variable_name + ', '+ series_type + ', ' + model + ', ' + experiment)
    plt.ylabel(variable_name + ' ['+y.units+']')
    plt.xlabel(series_type)

    
    
    
    
    
    
def plot_scatter(x,y,scatter_type):
    f, ax = plt.subplots(figsize = (12.5,8))
    res= stats.pearsonr(x,y)

    plt.scatter(x,y,facecolors='none', edgecolor='k')
    plt.ylabel(cloud_option + ' [' + y.units +']')
    plt.xlabel(org_option + ' ['+ x.units +']')
    title = cloud_option + ' and ' + org_option + ', ' + scatter_type + ', ' + model + ', ' + experiment

    if res[1]<=0.05:
        plt.title(title + ', R$^2$ = '+ str(round(res[0]**2,3)) + ', r=' + str(round(res[0],3)))
    else:
        plt.title(title + ', not statistically significant')









