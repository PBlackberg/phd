


import xarray as xr

da = xr.DataArray([1,2,3], dims = ['pev'], coords = {'pev':[1, 2, 3]})
print(da.dims)


# if 'plev' in da.dims or 'lev' in da.dims:
#     print('executes')


print('executes') if any(dim in da.dims for dim in ['plev', 'lev']) else None
    







