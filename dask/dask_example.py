


import numpy as np
import xarray as xr
import dask



a = xr.open_dataset('/Users/cbla0002/Documents/data/sample_data/pr/cmip6/ACCESS-CM2_pr_daily_historical_regridded.nc')['pr']
# print(a)
# print(a.dims)



# ------------------------------------------------------------------------------------ Chunking data --------------------------------------------------------------------------------------------------------- #
run = False
if run:
    import dask.array as da
    a = np.ones(shape = [10000, 10000, 1000]) # Takes a long time to load if at all
    print(a)           
    a = da.ones(shape = [10000, 10000, 1000])
    print(a)                                    # loads instantly
    x = da.random.random(size = (10000, 10000), chunks = (1000, 1000))
    print(x)
    x = da.random.random(size = (10000, 10000), chunks = (1000, 1000))
    y = x + x.T
    result = y.compute

run = False
if run:
    # a1 = a.chunk()
    # print(a1)

    a = a.chunk({'time': 366})
    print(a)

    chunks = a.chunks
    # print(chunks)
    
    total_chunks = 1
    for dim_chunks in chunks:
        total_chunks *= len(dim_chunks)
    print('number of chunks', total_chunks)

    a_mean = a.mean(dim = 'time')
    a_mean = a_mean.compute # There are 8 cores in a MAC pro. A core will calculate one chunk, and when it is finished it will go to the next (its good to make the chunks be a multiple of the available cores)
    print(a_mean)

    

# ------------------------------------------------------------------------------------ Scheduling calculation  --------------------------------------------------------------------------------------------------------- #
































