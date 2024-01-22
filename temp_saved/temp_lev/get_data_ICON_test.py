    # switch_test = {
    #     'test_file':    False,
    #     'test_month':   False,
    #     'test_year':    False,
    #     'test_full':    True
    #     }
    # test_get_data_ICON(switch_test, client) # code commented at the end


# def test_get_data_ICON(switch, client):    
#     print('getting ICON data started')
#     print(f'settings: {[key for key, value in switch.items() if value]}')
#     simulation_id, realm, frequency, variable_id  = 'ngc2013', 'atm', '3hour', 'pr'
#     path_files = iT.get_files_intake(simulation_id, frequency, variable_id, realm)
#     ds_grid, path_targetGrid = get_target_grid()
#     print(f'size of target grid dataset: {mFd.get_GiB(ds_grid)} GiB')
#     show = False
#     if show:
#         print('example of file structure:')
#         [print(file) for file in path_files[0:10]]
#         # [print(file) for file in path_files[-2::]]
#     if switch['test_file']:
#         print('getting sample')
#         ds = xr.open_mfdataset(path_files[0], combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
#         da = ds[variable_id]
#         print(f'dataset size: {mFd.get_GiB(ds)} GiB')
#         print(f'variable size: {mFd.get_MiB(da)} MiB')
#     elif switch['test_month']:
#         print('getting month')
#         ds = load_month(path_files)
#         da = ds[variable_id]
#         print(f'dataset size: {mFd.get_GiB(ds)} GiB') # ~ 126 GiB
#         print(f'variable size: {mFd.get_MiB(da)} MiB')
#         print(da)
#         da = da.resample(time="1D", skipna=True).mean()
#         da = mFd.persist_process(client, task = da, task_des = 'resample',persistIt = True, computeIt = True, progressIt = True)
#         print(da)
#         print(f'variable size after resampling: {mFd.get_MiB(da)} MiB')
#         da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
#         x_res, y_res = 2.5, 2.5
#         da_day = da.isel(time = 0)
#         print('da_day is here')
#         print(da_day)
#         path_gridDes, path_weights = rI.gen_weights(ds_in = da_day, x_res = x_res, y_res = y_res, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
#         ds = rI.remap(ds_in = da, path_gridDes = path_gridDes, path_weights = path_weights, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
#         print(f'this is the finished remapped and resampled, opened as chunnks')
#         print(ds)
#         print(f'variable size after remapping: {mFd.get_MiB(ds)} MiB')
#         # print(type(ds))
#         # exit()
#     elif switch['test_year']:
#         calc_weights = True
#         tempfiles_month = []
#         for month in np.arange(1, 13):
#             file_path = os.path.join(mV.folder_scratch, f'data_month_{month}.nc')
#             if os.path.exists(file_path):
#                 tempfiles_month.append(file_path)
#                 print(f"File for month {month} already exists. Skipping...")
#             else:
#                 print(f"Processing month: {month}")
#                 ds = load_month(path_files, month = month)
#                 da = ds[variable_id]
#                 da = da.resample(time="1D", skipna=True).mean()
#                 da = mFd.persist_process(client, task = da, task_des = 'resample',persistIt = True, computeIt = True, progressIt = True)
#                 da = da.assign_coords(clon=("ncells", ds_grid.clon.data * 180 / np.pi), clat=("ncells", ds_grid.clat.data * 180 / np.pi))
#                 if calc_weights:
#                     x_res, y_res = 2.5, 2.5
#                     path_gridDes, path_weights = rI.gen_weights(ds_in = da.isel(time = 0).compute(), x_res = x_res, y_res = y_res, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
#                     calc_weights = False
#                 ds_month = rI.remap(ds_in = da, path_gridDes = path_gridDes, path_weights = path_weights, path_targetGrid = path_targetGrid, path_scratch = mV.folder_scratch)
#                 # print(ds_month)
#                 # ds_year.append(ds_month)
#                 ds_month.to_netcdf(file_path)
#                 tempfiles_month.append(file_path)
#         # print(tempfiles_month)
#         ds = xr.open_mfdataset(tempfiles_month, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)
#         # combined_ds = xr.concat(ds_year, dim='time').sel(lat = slice(-30, 30))
#         # print(combined_ds)
#         # wait(combined_ds)
#         print(ds)
#         path_file = Path(mV.folder_scratch) / f"{variable_id}_2020_{int(360/x_res)}x{int(180/y_res)}.nc"
#         ds.to_netcdf(path_file, mode="w")
#         for temp_file in tempfiles_month:
#             os.remove(temp_file)


