# vars_highlight = None #'astring' #['a', 'b']
# print(type(vars_highlight))
# # exit()

# import xarray as xr
# vars_highlight = xr.Dataset(data_vars = {'a': xr.DataArray([1,2,3])})

# if vars_highlight is None:
#     print('executes')

# exit()
# if type(vars_highlight) == list:
#     print('executes')

# if type(vars_highlight) == int:
#     print('executes')

# if type(vars_highlight) == str:
#     print('executes')




# a = ['1', 'a', 'b']
# print('\n'.join(i for i in a))



# import numpy as np
# import matplotlib.pyplot as plt
# a = np.array([1, 2, 3])

# plt.figure()
# plt.plot(a)

# plt.show()



# import numpy as np
# import matplotlib.colors as mcolors

# def generate_distinct_colors(n):
#     hsv_colors = [(i / n, 1, 1) for i in range(n)]  # Hue varies, saturation and value are maxed
#     rgb_colors = [mcolors.hsv_to_rgb(color) for color in hsv_colors]
#     return rgb_colors

# # Example: Generate 5 distinct colors
# n_colors = 5
# colors = generate_distinct_colors(n_colors)
# print(colors)



    # file_pattern = "/Users/cbla0002/Desktop/pr/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_*.nc" 
    # paths = sorted(glob.glob(file_pattern))
    # da = xr.open_mfdataset(paths, combine='by_coords', parallel=True) # chunks="auto"




# switch = {'b' : False}
# if switch.get('b', False):
#     if switch['b']:
#         print('executes')


# import xarray as xr
# import numpy as np

# a = np.array([1, 4, 3])
# a = xr.DataArray(a)

# b = np.array([1, 2, np.nan])
# b = xr.DataArray(b)


# print((a/b).data)





# a = ['a', 'b']
# a = []

# if a:
#     print('executes')




# import functools
# import time
# import random

# def retry(exception_to_check, tries=4, delay=3, backoff=2, logger=None):
#     """
#     A retry decorator that allows a function to retry if a specific exception is caught.

#     Parameters:
#     - exception_to_check: Exception to check for retry.
#     - tries: Maximum number of attempts.
#     - delay: Initial delay between retries in seconds.
#     - backoff: Multiplier applied to delay each retry.
#     - logger: Logging.Logger instance for logging retries.

#     Usage:
#     @retry(ValueError, tries=5, delay=2, backoff=3)
#     def some_function_that_might_fail():
#         ...
#     """
#     def decorator_retry(func):
#         @functools.wraps(func)
#         def wrapper_retry(*args, **kwargs):
#             mtries, mdelay = tries, delay
#             while mtries > 1:
#                 try:
#                     return func(*args, **kwargs)
#                 except exception_to_check as e:
#                     if logger:
#                         logger.warning(f"Retry {tries-mtries+1}, waiting {mdelay} seconds: {e}")
#                     time.sleep(mdelay)
#                     mtries -= 1
#                     mdelay *= backoff
#             return func(*args, **kwargs)  # Last attempt
#         return wrapper_retry
#     return decorator_retry

# @retry(ValueError, tries=5, delay=2, backoff=2)
# def function_that_may_fail():
#     # Simulate a task that has a chance to fail
#     print("Attempting to perform a task...")
#     if random.randint(0, 1) == 0:
#         # Simulate a failure condition
#         raise ValueError("Simulated task failure")
#     else:
#         print("Task succeeded!")

# # Run the function to see the retry mechanism in action
# function_that_may_fail()


# import xarray as xr
# a = xr.open_dataset('/scratch/b/b382628/sample_data/pr/nextgems/ICON-ESM_ngc2013/ICON-ESM_ngc2013_pr_daily_historical_2024_regridded_144x72.nc')
# print(a)


# fractional frequency of occurence
# def calc_freq_occur(y):
#     ymin = np.nanmin(y)
#     ymax = np.nanmax(y)
#     edges = np.linspace(ymin, ymax, 101)
#     y_bins = []
#     for i in range(len(edges)-1):
#         if i == len(edges)-2:
#             y_value = (xr.where((y>=edges[i]) & (y<=edges[i+1]), 1, 0).sum() / len(y))*100
#         else:
#             y_value = (xr.where((y>=edges[i]) & (y<edges[i+1]), 1, 0).sum() / len(y))*100
#         y_bins.append(y_value)
#     bins_middle = edges[0:-1] + (edges[1] - edges[0])/2
#     return bins_middle, y_bins

# if switch_test['freq_occur']:
#     da = clwvi
#     _, y_bins = calc_freq_occur(da.values.flatten())
#     bins = np.arange(1,101)
#     ds[dataset] = xr.DataArray(y_bins)
#     ymin_foc.append(np.min(y_bins))
#     ymax_foc.append(np.max(y_bins))

# if switch_test['freq_occur']:
#     ymin = np.min(ymin_foc_list)
#     ymax = np.max(ymax_foc_list)
#     label_x = 'fraction of total range [%]'
#     label_y = 'Freq. occur [%]'
#     title = 'pe'
#     filename = f'{title}_value_distribution_fractional_values.png'
#     colors = lP.generate_distinct_colors(len(ds.data_vars))
#     fig, axes = lP.plot_dsLine(ds, x = bins, variable_list = list(ds.data_vars.keys()), title = filename, label_x = label_x, label_y = label_y, colors = colors, 
#                 ymin = ymin, ymax = ymax,
#                 fig_given = False, one_ax = True)
#     mP.show_plot(fig, show_type = 'save_cwd', filename = filename)




# import xarray as xr
# ds = xr.open_dataset('/scratch/w40/cb4968/sample_data/pr/cmip6/INM-CM5-0_pr_monthly_ssp585_regridded_144x72.nc')
# print(ds)





# # print(np.unique(conv_obj))
# # print(I)
# # print(J)
# # print(da)
# # exit()
# # print(labeli)
# mask = conv_obj == 1
# da = np.where(mask, 1, 0)
# print(da)
# print(np.unique(da))
# fig = plt.figure()
# plt.imshow(da, cmap='viridis')
# # plt.show()
# plt.savefig("zome_plots/convex_polygtope_solution.png")
# exit()



# print(result)
# exit()
# for i in range(14):
#     print(f'lat1{np.unique(lat1[:,:,i])}')
#     print(f'lon1{np.unique(lon1[:,:,i])}')
# exit()


# import warnings
# with warnings.catch_warnings():
#     warnings.filterwarnings('error', 'invalid value encountered in arcsin')
#     try:
#         result = 2 * R * np.arcsin(np.sqrt(h))
#     except RuntimeWarning:
#         print("Warning caught: invalid value encountered in arcsin")
#         print(np.min(h))
#         print(np.max(h))
#         exit()



#[None] * int(np.math.factorial(len(labels)) / (2 * np.math.factorial((len(labels)-2))))


            #     print(f'label_i:{label_i.data}')
            #     print(f'label_j:{label_j.data}')
            #     ds_obj_dist = xr.Dataset()
            #     distance_from_obj_i = distance_from_obj_i.where(distance_from_obj_i > 0)
            #     ds_obj_dist['object'] = distance_from_obj_i
            #     ds = ds_obj_dist
            #     label = 'convection [0,1]'
            #     vmin = None
            #     vmax = None
            #     cmap = 'Blues_r'
            #     filename = f'distance_from_{i}.png'
            #     fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            #     mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

            #     ds_obj_dist = xr.Dataset()
            #     distance_from_obj_i = distance_from_obj_i.where(distance_from_obj_i > 0)
            #     scene_j_nan = xr.where(scene_j>0, np.nan, 0)
            #     ds_obj_dist['object'] = distance_from_obj_i + scene_j_nan
            #     ds = ds_obj_dist
            #     label = 'convection [0,1]'
            #     vmin = None
            #     vmax = None
            #     cmap = 'Blues_r'
            #     filename = f'obj_{j}.png'
            #     fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            #     mP.show_plot(fig, show_type = 'save_cwd', filename = filename)
            #     exit()

            # if i == 8 and j == 0:
            #     ds_obj_dist = xr.Dataset()
            #     distance_from_obj_i = distance_from_obj_i.where(distance_from_obj_i > 0, np.nan)
            #     ds_obj_dist['object'] = distance_from_obj_i
            #     ds = ds_obj_dist
            #     label = 'convection [0,1]'
            #     vmin = None
            #     vmax = None
            #     cmap = 'Blues_r'
            #     filename = f'distance_from_{i}.png'
            #     fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            #     mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

            #     ds_obj_dist = xr.Dataset()
            #     distance_from_obj_i = distance_from_obj_i.where(distance_from_obj_i > 0, np.nan)
            #     scene_j_nan = xr.where(scene_j>0, np.nan, 0)
            #     ds_obj_dist['object'] = distance_from_obj_i + scene_j_nan
            #     ds = ds_obj_dist
            #     label = 'convection [0,1]'
            #     vmin = None
            #     vmax = None
            #     cmap = 'Blues_r'
            #     filename = f'obj_{j}.png'
            #     fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
            #     mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

            #     print(A_a.data)
            #     print(A_b.data)
            #     print(np.sqrt(A_d.data))
            #     exit()


        # x = da_mean_lat.values
        # y = da_mean_lat['lat'].values
        # fig = plt.figure(figsize=(8, 6))
        # plt.plot(x, y, 'k')
        # plt.xlabel('time-lon mean wap')
        # plt.ylabel('lat')
        # plt.axvline(x=0, color='k', linestyle='--')
        # plt.axhline(y=0, color='k', linestyle='--')
        # mFp.show_plot(fig, show_type = 'save_cwd', filename = f'{dataset}_{experiment}_{mV.resolutions[0]}_wap{height}_tMean_lonMean')
        
        
            


# import xarray as xr
# ds = xr.open_dataset('/scratch/w40/cb4968/sample_data/conv_obj/cmip6/TaiESM1_conv_obj_daily_historical_regridded_144x72.nc')
# print(ds)

