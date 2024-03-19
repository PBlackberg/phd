'''
# --------------------
#  parallel_calc_test
# --------------------
'''


import numpy as np
import xarray as xr
import dask
from distributed import Client
import time

def create_client():
    client = Client()
    print(client)
    print(f"Dask Dashboard is available at {client.dashboard_link}")
    import webbrowser
    webbrowser.open(f'{client.dashboard_link}') 
    return client

@dask.delayed
def my_function_dask_delayed(x):
    time.sleep(1)
    return (x ** 2).data

def my_function(x):
    time.sleep(1)
    return (x ** 2).data

@dask.delayed
def process_data(data):
    time.sleep(4)
    return data.sum()


if __name__ == '__main__':
    client = create_client()

    switch = {
        'test_parallel_calc':   False,
        'serial_calc':          False,
        'parallel_calc':        False,
        'test_client_scatter':  True
        }

    # ----------------------------------------------------------------- get_data ------------------------------------------------------------------------------ #
    aList = np.arange(1, 10)
    aList_xr = xr.DataArray(aList)
    # print(aList)


    # ----------------------------------------------------------------- calculate ------------------------------------------------------------------------------ #
    if switch['test_parallel_calc']:
        result = [my_function_dask_delayed(i) for i in aList_xr]
        print(result)
        result = dask.compute(result)
        print(result)

    if switch['serial_calc']:
        start_time = time.time()
        result = [my_function(i) for i in aList_xr]
        serial_time = time.time() - start_time
        print(f'serial computation took:{np.round(serial_time, 2)}')

    if switch['parallel_calc']:
        start_time = time.time()
        result = [my_function_dask_delayed(i) for i in aList_xr]
        result = dask.compute(result)
        print(result)
        print(type(result))
        parallel_time = time.time() - start_time
        print(f'parallel computation took:{np.round(parallel_time, 2)}')

    if switch['test_client_scatter']:
        large_dataset = np.random.rand(1000000)
        future = client.scatter(large_dataset, direct=True, broadcast=True)
        result_future = process_data(future)
        result = result_future.compute()
        print(result)

    client.close()











# alternative way
# @dask.delayed
# def subset_year(conv_obj_year, obj_id_masked_year):
#     subset_year = [subset_day(conv_obj_year, obj_id_masked_year, day) for day in range(len(conv_obj_year.time))]
#     return subset_year

# def get_obj_subset(conv_obj, obj_id, metric_id_mask):
#     ''' 
#     conv_obj:      (time, lat, lon) (integer values)
#     obj_id:        (time, obj)      (integer values and NaN)
#     metric_mask:   (time, obj)      ([NaN, 1])
#     '''
#     obj_id_masked = obj_id * metric_id_mask
#     # conv_obj = conv_obj.chunk({'time':'auto'})
#     # print(conv_obj)
#     # conv_obj_copies = client.scatter(conv_obj, direct=True, broadcast=True)
#     # obj_id_masked_copies = client.scatter(obj_id_masked, direct=True, broadcast=True)
#     futures = []
#     for year in np.unique(conv_obj['time'].dt.year):
#         conv_obj_year = conv_obj.sel(time = f'{year}')
#         obj_id_masked_year = obj_id_masked.sel(time = f'{year}')
#         futures.append(subset_year(conv_obj_year, obj_id_masked_year))
#     result = dask.compute(futures)[0]
#     result = list(itertools.chain.from_iterable(result))
#     conv_obj_subset = xr.concat(result, dim='time')
#     print(conv_obj_subset)
#     return conv_obj_subset










