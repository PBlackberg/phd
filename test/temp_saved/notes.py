# for cancat files:
# for f in files:  # one model from warming scenario from cmip5 have a file that needs to be removed (creates duplicate data otherwise)
#     files.remove(f) if f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]=='19790101' and f[f.index(".nc")-fileYear2_charStart : f.index(".nc")]=='20051231' else None





# settings for plot with individual numbers on xaxis for trend plot
        # mF.move_row(ax, 0.0875 - 0.025 +0.025) if row == 0 else None
        # mF.move_row(ax, 0.0495 - 0.0135+0.025) if row == 1 else None
        # mF.move_row(ax, 0.01   - 0.005+0.025)  if row == 2 else None
        # mF.move_row(ax, -0.0195+0.025)         if row == 3 else None
        # mF.move_row(ax, -0.05+0.025)           if row == 4 else None
        # mF.move_row(ax, -0.05+0.025)           if row == 5 else None



# creating a mask for excluding imcomplete sections
# da = gD.get_var_data(source, dataset, experiment, 'ta', switch)
# nan_mask = da.isnull().any(dim='plev')                        
# da = da.where(~nan_mask, np.nan)                              # When calculating the difference between two sections I exclude gridpoints where the value at any included pressure level is NaN
# theta =  da * (1000e2 / da.plev)**(287/1005)                  # theta = T (P_0/P)^(R_d/C_p)
# plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]               # pressure levels in ERA are reversed to cmip
# da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))] if not dataset == 'ERA5' else [theta.sel(plev=slice(plevs1[1], plevs1[0])), theta.sel(plev=slice(plevs2[1], plevs2[0]))] 
# da = ((da1 * da1.plev).sum(dim='plev') / da1.plev.sum(dim='plev')) - ((da2 * da2.plev).sum(dim='plev') / da2.plev.sum(dim='plev'))   











# previous geeting cmip5 and cmip6 separately
# ------------------------------------------------------------------------------- For most variables ----------------------------------------------------------------------------------------------------------#
# def get_cmip5_data(variable, model, experiment, switch = {'ocean': False}):
#     ''' concatenates file data and interpolates grid to common grid if needed '''    
#     ensemble = choose_cmip5_ensemble(model, experiment)
#     timeInterval = ('day', 'day') if mV.timescales[0] == 'daily' else ('mon', 'Amon')
#     path_gen = f'/g/data/al33/replicas/CMIP5/combined/{mV.institutes[model]}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
#     version = latestVersion(path_gen)
#     path_folder = f'{path_gen}/{version}/{variable}'
#     ds = concat_files(path_folder, variable, model, experiment)
#     da= ds[variable]
#     if mV.resolutions[0] == 'regridded':
#         import regrid_xesmf as regrid
#         regridder = regrid.regrid_conserv_xesmf(ds)
#         da = regridder(da)
#     ds = xr.Dataset(data_vars = {f'{variable}': da.sel(lat=slice(-30,30))}, attrs = ds.attrs)
#     return ds


# # --------------------------------------------------------------------------------- For cloudfraction ----------------------------------------------------------------------------------------------------------#
# def get_cmip5_cl(variable, model, experiment, switch = {'ocean': False}):
#     ds_cl, ds_p_hybridsigma = None, None
#     ''' Cloud pressure on hybrid-sigma vertical levels '''    
#     ensemble = choose_cmip5_ensemble(model, experiment)
#     timeInterval = ('day', 'day') if mV.timescales[0] == 'daily' else ('mon', 'Amon')
#     path_gen = f'/g/data/al33/replicas/CMIP5/combined/{mV.institutes[model]}/{model}/{experiment}/{timeInterval[0]}/atmos/{timeInterval[1]}/{ensemble}'
#     version = latestVersion(path_gen)
#     path_folder = f'{path_gen}/{version}/{variable}'
#     ds = concat_files(path_folder, variable, model, experiment)
        

    
#     if mV.resolutions[0] == 'orig':
#         ds_cl = ds # units in % on sigma pressure coordinates
#         ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
#     if mV.resolutions[0] == 'regridded':
#         import regrid_xesmf as regrid
#         cl = ds['cl'] # units in % on sigma pressure coordinates
#         regridder = regrid.regrid_conserv_xesmf(ds)
#         cl = regridder(cl)
#         p_hybridsigma_n = regridder(p_hybridsigma)
#         ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
#         ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma_n}, attrs = ds.lev.attrs)
#     return ds_cl, ds_p_hybridsigma

# --------------------------------------------------------------------------------- For cloudfraction ----------------------------------------------------------------------------------------------------------#
# def get_cmip6_cl(variable, model, experiment, switch = {'ocean': False}):
#     ''' Cloud pressure on hybrid-sigma vertical levels '''
#     ensemble = choose_cmip6_ensemble(model, experiment)
#     project = 'CMIP' if experiment == 'historical' else 'ScenarioMIP'
#     timeInterval = 'day' if mV.timescales[0] == 'daily' else 'Amon'

#     if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
#         path_gen = f'/g/data/fs38/publications/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'
#     else:
#         path_gen = f'/g/data/oi10/replicas/CMIP6/{project}/{mV.institutes[model]}/{model}/{experiment}/{ensemble}/{timeInterval}/{variable}'

#     # print(path_gen)
#     folder_grid = grid_folder(model)
#     version = latestVersion(os.path.join(path_gen, folder_grid)) if not model in ['ACCESS-ESM1-5', 'ACCESS-CM2'] else 'latest'
#     path_folder =  f'{path_gen}/{folder_grid}/{version}'
#     ds = concat_files(path_folder, experiment) # picks out lat: [-35, 35]
#     if model == 'IITM-ESM': # different models have different conversions from height coordinate to pressure coordinate.
#         ds = ds.rename({'plev':'lev'})
#         p_hybridsigma = ds['lev'] # already on pressure levels
#     elif model == 'IPSL-CM6A-LR':
#         ds = ds.rename({'presnivs':'lev'})
#         p_hybridsigma = ds['lev'] # already on pressure levels
#     elif model in ['MPI-ESM1-2-HR', 'MPI-ESM1-2-LR', 'CanESM5', 'CNRM-CM6-1', 'GFDL-CM4', 'CNRM-ESM2-1', 'CNRM-CM6-1-HR']: 
#         p_hybridsigma = ds.ap + ds.b*ds.ps
#     elif model == 'FGOALS-g3':
#         p_hybridsigma = ds.ptop + ds.lev*(ds.ps-ds.ptop)
#     elif model in ['UKESM1-0-LL', 'KACE-1-0-G', 'ACCESS-CM2', 'ACCESS-ESM1-5']:
#         h_hybridsigma = ds.lev + ds.b*ds.orog                                                      # in meters
#         p_hybridsigma = 1000e2 * (1 -  0.0065*h_hybridsigma/288.15)**(9.81*0.029)/(8.314*0.0065) # to pressure: P = P_0 * (1- L*(h-h_0)/T_0)^(g*M/R*L) Barometric formula (approximation based on lapserate)
#         # p_hybridsigma = 1000e2 * np.exp(0.029*9.82*h_hybridsigma/287*T)                        # to pressure: P = P_0 * exp(- Mgh/(RT)) Hydrostatic balance (don't have T at pressure level)
#     else:
#         p_hybridsigma = ds.a*ds.p0 + ds.b*ds.ps

#     if mV.resolutions[0] == 'orig':
#         ds_cl = ds # units in % on sigma pressure coordinates
#         ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
#     if mV.resolutions[0] == 'regridded':
#         import regrid_xesmf as regrid
#         cl = ds['cl'] # units in % on sigma pressure coordinates
#         regridder = regrid.regrid_conserv_xesmf(ds)
#         cl = regridder(cl)
#         ds_cl = xr.Dataset(data_vars = {'cl': cl}, attrs = ds.attrs)
#         p_hybridsigma = regridder(p_hybridsigma) if not model in ['IITM-ESM', 'IPSL-CM6A-LR'] else p_hybridsigma
#         ds_p_hybridsigma = xr.Dataset(data_vars = {'p_hybridsigma': p_hybridsigma}, attrs = ds.lev.attrs)
#     return ds_cl, ds_p_hybridsigma






# --------------------------------------------------------------- Taking mean of levels falling in fixed ranges (height and then pressure) (not used) ----------------------------------------------------------------------------------------------------- #
# @mF.timing_decorator
# def convert_to_p(da, z, z_new):
#     print(f'Column cloud fraction value before interp pressure conversion:\n {da.isel(time=0, lat = int(len(ds.lat)/2), lon = 5).values} \n at coordinate: \n {da.lev.values}')
#     print(f'at height coordinates: \n', z.isel(lat = int(len(ds.lat)/2), lon = 5).values)
#     interpolated_data = np.zeros(shape = (len(da['time']), len(z_new), len(da['lat']), len(da['lon'])))
#     da_z_fixed = xr.DataArray(interpolated_data, dims = ('time', 'lev', 'lat', 'lon'), coords = {'time':da.time.data, 'lev': z_new.data, 'lat': da.lat.data, 'lon': da.lon.data})
#     for i in range(len(z_new)-1):
#         z_section = z.where((z >= z_new[i]) & (z < z_new[i+1]))                   # values not in the section are NaN
#         da_z_section = da * ~np.isnan(z_section)                                  # cloudfraction and NaNs
#         da_z_slice = da_z_section.where(da_z_section != 0).mean(dim='lev')                                 # .where(da_z_section != 0)
#         da_z_fixed[:, i+1] = da_z_slice                                           # First level is zero m, so assigning mean values to level above as first value
#     da_z_fixed[:, 0] = np.nan
#     print(f'Column cloud fraction value after interp pressure conversion:\n {da_z_fixed.isel(time=0, lat = int(len(ds.lat)/2), lon = 5).values}, \n interp heights: \n {da_z_fixed.lev.values}') #int(len(ds.lon)/2)
#     da_z_fixed['lev'] = 101325*(1-((2.25577e-5)*da_z_fixed['lev']))**(5.25588)    # convert new heights to pressure
#     da_z_fixed = da_z_fixed.rename({'lev':'plev'})
#     print('interp pressure levels: \n', da_z_fixed.plev.values)
#     return da_z_fixed

# @mF.timing_decorator
# def interp_vert_p(da, p_new):   
#     interpolated_data = np.zeros(shape = (len(da['time']), len(p_new), len(da['lat']), len(da['lon'])))
#     da_p_fixed = xr.DataArray(interpolated_data, dims = ('time', 'plev', 'lat', 'lon'), coords = {'time':da.time.data, 'plev': p_new.data, 'lat': da.lat.data, 'lon': da.lon.data})
#     for i in range(len(p_new)-1):
#         plevs = da.plev
#         da_p_section = da.where((plevs <= p_new[i]) & (plevs > p_new[i+1]))
#         da_p_slice = da_p_section.mean(dim='plev')
#         da_p_fixed[:, i] = da_p_slice
#     da_p_fixed[:, -1] = np.nan
#     print(f'Column cloud fraction after pressure interp:\n {da_p_fixed.isel(time=0, lat = int(len(ds.lat)/2), lon = int(len(ds.lon)/2)).values}, \n interp pressures: \n {da_p_fixed.plev.values}')

#     calc_by_mean = False
#     if calc_by_mean:
#         da = convert_to_p(da, z, z_new) # takes: ~ 2 min
#         da = interp_vert_p(da, p_new)   # takes: ~ 0.15 min



# pressure level conversion
# plevs = 1000e2 * (1 -  0.0065*h_hybridsigma/288.15)**(9.81*0.029)/(8.314*0.0065)         # to pressure: P = P_0 * (1- L*(h-h_0)/T_0)^(g*M/R*L) Barometric formula (approximation based on lapserate)
# p_hybridsigma = 1000e2 * np.exp(0.029*9.82*h_hybridsigma/287*T)                        # to pressure: P = P_0 * exp(- Mgh/(RT)) Hydrostatic balance (don't have T at pressure level)
# da_h_new = da.interp(plev=plevs, new_z=h_new, method='linear')
# da_p_new = da.interp(plev=plevs, new_z=p_new, method='linear')







# previous way of getting low clouds and high clouds
        # elif var in ['lcf', 'hcf']:
        #     p_hybridsigma = gD.get_var_data(source, dataset, experiment, 'p_hybridsigma', switch)
        #     da = gD.get_var_data(source, dataset, experiment, 'cl', switch)
        #     plevs1, plevs2 = [250e2, 0], [1500e2, 600e2]
        #     da = da.where((p_hybridsigma <= plevs1[0]) & (p_hybridsigma >= plevs1[1]), 0).max(dim='lev') if var == 'hcf' else da
        #     da = da.where((p_hybridsigma <= plevs2[0]) & (p_hybridsigma >= plevs2[1]), 0).max(dim='lev') if var == 'lcf' else da