import numpy as np


# ---------------------------------------------------------------------------- functions for common operations --------------------------------------------------------------------------------------------------- #

def snapshot(da):
    ''' Take a snapshot from a timestep of the dataset '''
    return da.isel(time=0)

def tMean(da):
    ''' Calculate time-mean '''
    return da.mean(dim='time', keep_attrs=True)

def sMean(da):
    ''' Calculate area-weighted spatial mean '''
    aWeights = np.cos(np.deg2rad(da.lat))
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

def mean(da):
    ''' Calculate spatial mean, then time-mean '''
    return tMean(sMean(da))

def get_super(x):
    ''' For adding superscripts in strings (input is string) '''
    normal = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+-=()"
    super_s = "ᴬᴮᶜᴰᴱᶠᴳᴴᴵᴶᴷᴸᴹᴺᴼᴾQᴿˢᵀᵁⱽᵂˣʸᶻᵃᵇᶜᵈᵉᶠᵍʰᶦʲᵏˡᵐⁿᵒᵖ۹ʳˢᵗᵘᵛʷˣʸᶻ⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾"
    res = x.maketrans(''.join(normal), ''.join(super_s))
    return x.translate(res)






# -------------------------------------------------------------------------------------- functions for scenes --------------------------------------------------------------------------------------------------- #

def connect_boundary(da):
    ''' Connect objects across boundary 
    Objects that touch across lon=0, lon=360 boundary are the same object.
    Takes array(lat, lon)) 
    '''
    s = np.shape(da)
    for row in np.arange(0,s[0]):
        if da[row,0]>0 and da[row,-1]>0:
            da[da==da[row,0]] = min(da[row,0],da[row,-1])
            da[da==da[row,-1]] = min(da[row,0],da[row,-1])

def haversine_dist(lat1, lon1, lat2, lon2):
    '''Great circle distance (from Haversine formula) (used for distance between objects)
    h = sin^2(phi_1 - phi_2) + (cos(phi_1)cos(phi_2))sin^2(lambda_1 - lambda_2)
    (1) h = sin(theta/2)^2
    (2) theta = d_{great circle} / R    (central angle, theta)
    (1) in (2) and rearrange for d gives
    d = R * sin^-1(sqrt(h))*2 

    where 
    phi -latitutde
    lambda - longitude
    (Takes vectorized input)
    '''
    R = 6371 # radius of earth in km
    lat1 = np.deg2rad(lat1)                       
    lon1 = np.deg2rad(lon1-180) # function requires lon [-180 to 180]
    lat2 = np.deg2rad(lat2)                       
    lon2 = np.deg2rad(lon2-180)
    
    h = np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2) * np.sin((lon2 - lon1)/2)**2 # Haversine formula
    return 2 * R * np.arcsin(np.sqrt(h))





# ------------------------------------------------------------------------------------- functions for time series --------------------------------------------------------------------------------------------------- #

def monthly_clim(da):
    ''' Creates a data array with the climatology of each month  '''
    year = da.time.dt.year
    month = da.time.dt.month
    da = da.assign_coords(year=("time", year.data), month=("time", month.data))
    return da.set_index(time=("year", "month")).unstack("time") # reshape the array to ("month", "year")

def resample_timeMean(da, timeMean_option=''):
    ''' Resample data to specified timescale [annual, seasonal, monthly, daily]'''
    if timeMean_option == 'annual' and len(da) >= 100:
        da = da.resample(time='Y').mean(dim='time', keep_attrs=True)

    elif timeMean_option == 'seasonal' and len(da) >= 100:
        da = da.resample(time='QS-DEC').mean(dim="time")
        da = monthly_clim(da)
        da = da.rename({'month':'season'})
        da = da.assign_coords(season=["MAM", "JJA", "SON", "DJF"])
        da = da.isel(year=slice(1, None))

    elif timeMean_option == 'monthly' and len(da) > 360:
        da = da.resample(time='M').mean(dim='time', keep_attrs=True)

    elif timeMean_option == 'daily' or not timeMean_option:
        pass
    return da


    









# --------------------------------------------------------------------------   saved for the moment   ---------------------------------------------------------------------------------- #


# def get_dsvariable(variable, dataset, experiment = 'historical', home = os.path.expanduser("~") + '/Documents', resolution='regridded', timescale = 'monthly'):
#     ''' Load saved variable '''
#     folder = '{}/data/CMIP5/ds_cmip5_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + variable + '_' + timescale +'_' + experiment + '_' + resolution + '.nc'
#     path_cmip5 = os.path.join(folder, filename)
#     try:
#         ds = xr.open_dataset(path_cmip5)
#     except FileNotFoundError:

#         try:
#             folder = '{}/data/CMIP6/ds_cmip6_{}/{}'.format(home, resolution, dataset)
#             filename = dataset + '_' + variable + '_' + timescale +'_' + experiment + '_' + resolution + '.nc'
#             path_cmip6 = os.path.join(folder, filename)
#             ds = xr.open_dataset(path_cmip6)
#         except FileNotFoundError:

#             try:
#                 folder = '{}/data/obs/ds_obs_{}/{}'.format(home, resolution, dataset)
#                 filename = dataset + '_' + variable + '_' + timescale +'_' + resolution + '.nc'
#                 path_obs = os.path.join(folder, filename)
#                 ds = xr.open_dataset(path_obs)
#             except FileNotFoundError:
#                 print(f"Error: no file at {path_cmip5}, {path_cmip6}, or {path_obs}")
#     return ds





# def get_metric(metric, dataset, experiment='historical', home=os.path.expanduser("~") + '/Documents', resolution='regridded'):
#     ''' load   '''

#     folder = '{}/data/CMIP5/metrics_cmip5_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + metric + '_' + experiment + '_' + resolution + '.nc'
#     path_cmip5 = os.path.join(folder, filename)

#     folder = '{}/data/CMIP6/metrics_cmip6_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + metric + '_' + experiment + '_' + resolution + '.nc'
#     path_cmip6 = os.path.join(folder, filename)

#     folder = '{}/data/obs/ds_obs_{}/{}'.format(home, resolution, dataset)
#     filename = dataset + '_' + metric + '_' + resolution + '.nc'
#     path_obs = os.path.join(folder, filename)

#     try:
#         ds = xr.open_dataset(path_cmip5)
#     except FileNotFoundError:
#         try:
#             ds = xr.open_dataset(path_cmip6)
#         except FileNotFoundError:
#             try:
#                 ds = xr.open_dataset(path_obs)
#             except FileNotFoundError:
#                 print(f"Error: no file at {path_cmip5}, {path_cmip6}, or {path_obs}")
#     return ds






# for experiment in experiments:
#     if experiment and source in ['cmip5', 'cmip6']:
#         print(f'\t {experiment}') if pD.prData_exist(dataset, experiment) else print(f'\t no {experiment} data')
#     print( '\t obserational dataset') if not experiment and source == 'obs' else None

#     if mV.no_data(source, experiment, pD.prData_exist(dataset, experiment)):
#         continue

#     da = load_data(switch, dataset, experiment, folder_save, timescale, resolution)
#     calc_metrics(switch, da, folder_save, source, dataset, experiment)





# def pick_region(da, dataset, experiment = 'historical', region = 'total', timeMean_option='monthly'):
#     ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
#     wap500 = get_wap(dataset, experiment)['wap'].sel(plev = 500e2)
#     if 'time' in da.dims:
#         wap500 = resample_timeMean(wap500, timeMean_option)
#         wap500 = wap500.assign_coords(time=da.time)
#     else: 
#         wap500 = wap500.mean(dim='time') # when looking at variables in regions of mean descent
#     if region == 'total':
#         pass
#     elif region == 'descent':
#         da = da.where(wap500>0)
#     elif region == 'ascent':
#         da = da.where(wap500<0)
#     return da

# def pick_region(data, dataset, experiment = 'historical', region = 'descent'):
#     ''' Picks out total region, region of descent, or region of ascent based on vertical pressure velocity at 500 hPa'''
#     wap = get_dsvariable('wap500', dataset, experiment)['wap500']
#     if 'time' in data.dims:
#         wap = wap.assign_coords(time=data.time)
#     else:
#         wap = wap.mean(dim='time')

#     if region == 'total':
#         pass
#     elif region == 'descent':
#         data = data.where(wap>0)
#     elif region == 'ascent':
#         data = data.where(wap<0)
#     return data


