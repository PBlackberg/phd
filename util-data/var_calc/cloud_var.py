






def get_clouds(switch, var_name, dataset, experiment, resolution, timescale):
    ''' # can also do 250 up (in schiro 'spread paper') '''
    da = vB.load_variable({'cl':True}, switch, dataset, experiment, resolution, timescale)
    da = da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev') if var_name == 'lcf' else da
    da = da.sel(plev = slice(400e2, 0)).max(dim = 'plev')      if var_name == 'hcf' else da  
    return da






