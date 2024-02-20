# folder_variableParent = Path(mV.folder_scratch) / variable_id
# folder_variableParent.mkdir(exist_ok=True)
# folder_variable = Path(folder_variableParent) / model
# folder_variable.mkdir(exist_ok=True)



# def test():
#     if switch['test_sample']:
#         years_range = [2020, 2020]
#     folder_dataProcessed = f'{sF.folder_scratch}/sample_data/{var_name}/{source}/{model}'
#     if os.path.exists(folder_dataProcessed):
#         print(f'getting {model} daily {var_name} data at {x_res}x{y_res} deg, between {years_range[0]}:{years_range[1]}')
#         path_dataProcessed = [os.path.join(folder_dataProcessed, f) for f in os.listdir(folder_dataProcessed) if os.path.isfile(os.path.join(folder_dataProcessed, f))]
#         if path_dataProcessed == []:
#             print(f'no {model}: daily {var_name} data at {x_res}x{y_res} deg in {folder_dataProcessed} (check)')
#         path_dataProcessed.sort()

#         if switch['test_sample']:
#             ds.coords['lon'] = ds.coords['lon'] + 180
#             return xr.open_mfdataset(path_dataProcessed[0], combine="by_coords", chunks="auto", engine="netcdf4", parallel=True).sel(lat = slice(-30, 30))*24*60*60 # open first processed year       
#         ds = xr.open_mfdataset(path_dataProcessed, combine="by_coords", chunks="auto", engine="netcdf4", parallel=True)        # open all processed years
#         years_ds = ds["time"].dt.year.values
#         if years_ds[0] != years_range[0] or years_ds[-1] != years_range[1]:
#             print(f'years requested: {years_range[0]}:{years_range[-1]}')
#             print(f'time range of processed {var_name} data is {years_ds[0]}:{years_ds[-1]}')
#             response = request_process(model, var_name, years_range)
#             if response == 'y':
#                 del ds
#                 process_data(model, var_name, x_res, y_res, years_range)
#             if response == "n":
#                 print('returning existing years')
#                 return ds[var_name]
            
#         else:
#             ds.coords['lon'] = ds.coords['lon'] + 180
#             return ds[var_name].sel(lat = slice(-30, 30))*24*60*60
#     else:
#         print(f'no {model}: daily {var_name} data in {folder_dataProcessed}')
#         response = request_process(model, var_name, years_range)
#         if response == 'y':
#             process_data(model, var_name, x_res, y_res, years_range)