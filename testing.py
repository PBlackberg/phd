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


