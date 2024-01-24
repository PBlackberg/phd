'''
# ------------------------
#   Large-scale state
# ------------------------
This script calculates spatial and temporal means of key variables
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr

# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                                        
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myVars as mV                                 
import myFuncs as mF     



# ------------------------
#       Get data
# ------------------------
# -------------------------------------------------------------------------------------- pick region ----------------------------------------------------------------------------------------------------- #
def calc_vMean(da, plevs0 = 850e2, plevs1 = 0):         # free troposphere (as most values at 1000 hPa and 925 hPa over land are NaN)                                            #  
    da = da.sel(plev = slice(plevs0, plevs1))
    w = ~np.isnan(da) * da['plev']                      # Where there are no values, exclude the associated pressure levels from the weights
    da = (da * w).sum(dim='plev') / w.sum(dim='plev') 
    return da

def pick_vert_reg(switch, da):
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        da, region = [da.sel(plev = 700e2), '_700hpa']  if met_type == '700hpa' and 'plev' in da.dims   else [da, region]
        da, region = [da.sel(plev = 500e2), '_500hpa']  if met_type == '500hpa' and 'plev' in da.dims   else [da, region]
        da, region = [da.sel(plev = 250e2), '_250hpa']  if met_type == '250hpa' and 'plev' in da.dims   else [da, region]
        da, region = [calc_vMean(da), 'vMean']          if met_type == 'vMean'  and 'plev' in da.dims   else [da, region]
    return da, region

def pick_hor_reg(switch, dataset, experiment, da):
    ''' Ascent/descent region based on 500 hPa vertical pressure velocity (wap)'''
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        wap500 = mF.load_variable(switch, 'wap', dataset, experiment).sel(plev = 500e2) if ('descent' in switch and met_type == 'descent') or ('ascent' in switch and met_type == 'ascent') else None
        da, region = [da.where(wap500 > 0), '_d']  if met_type == 'descent' else [da, region]
        da, region = [da.where(wap500 < 0), '_a']  if met_type == 'ascent'  else [da, region]

        wap500 = mF.load_variable(switch, 'wap', dataset, experiment).sel(plev = 500e2).mean(dim='time') if ('descent_fixed' in switch and met_type == 'descent_fixed') or ('ascent_fixed' in switch and met_type == 'ascent_fixed') else None
        da, region = [da.where(wap500 > 0), '_fd']  if met_type == 'descent_fixed' else [da, region]
        da, region = [da.where(wap500 < 0), '_fa']  if met_type == 'ascent_fixed'  else [da, region]
        region = f'_o{region}'                      if met_type == 'ocean' else region   # loading data deals with picking out ocean (as it can be done before or after interpolation)                         
    return da, region


# -------------------------------------------------------------------------------------- Get variable ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator()
def get_variable(switch, var, dataset, experiment):
    print(f'getting {var}')
    if var in ['lcf', 'hcf']:
        da = mF.load_variable({'cl':True}, switch, dataset, experiment)
        da = da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev') if var == 'lcf' else da
        da = da.sel(plev = slice(400e2, 0)).max(dim = 'plev')      if var == 'hcf' else da  # can also do 250 up (in schiro 'spread paper')
    elif var == 'stability':                                                                # Differnece in potential temperature between two vertical sections
        da = mF.load_variable({'ta': True}, switch, dataset, experiment)                    # Temperature at pressure levels (K)
        theta =  da * (1000e2 / da['plev'])**(287/1005) 
        plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
        da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
        w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']                 # Where there are no temperature values, exclude the associated pressure levels from the weights
        da = ((da1 * w1).sum(dim='plev') / w1.sum(dim='plev')) - ((da2 * w2).sum(dim='plev') / w2.sum(dim='plev'))
    elif var == 'netlw':    
        rlds, rlus, rlut = [mF.load_variable({var: True}, switch, dataset, experiment) for var in ['rlds', 'rlus', 'rlut']]
        da = -rlds + rlus - rlut
    elif var == 'netsw':           
        rsdt, rsds, rsus, rsut = [mF.load_variable({var: True}, switch, dataset, experiment) for var in ['rsdt', 'rsds', 'rsus', 'rsut']]
        da = rsdt - rsds + rsus - rsut
    elif var == 'h':                                                                        # h - Moist Static Energy (MSE)
        c_p, L_v = mF.dims_class.c_p, mF.dims_class.L_v
        ta, zg, hus = [mF.load_variable({var: True}, switch, dataset, experiment) for var in ['ta', 'zg', 'hus']]
        da = c_p * ta + zg + L_v * hus
    elif var == 'h_anom2':                                                                  # MSE variance from the tropical mean
        c_p, L_v = mF.dims_class.c_p, mF.dims_class.L_v
        ta, zg, hus = [mF.load_variable({var: True}, switch, dataset, experiment) for var in ['ta', 'zg', 'hus']]
        da = c_p * ta + zg + L_v * hus
        da, _ = pick_vert_reg(switch, dataset, da)
        da_sMean = get_sMean(da)
        da_anom = da - da_sMean
        da = da_anom**2
        # plot_object = da.isel(time=0).plot()
        # fig = plot_object.figure
        # fig.savefig(f'{os.getcwd()}/test/plot_test/test2.png')
    elif var == 'pe':
        pr = mF.load_variable({'pr': True}, switch, dataset, experiment, timescale = 'daily').resample(time='1MS').mean(dim='time')    # mm/m^2/day
        clwvi = mF.load_variable({'clwvi': True}, switch, dataset, experiment).resample(time='1MS').mean(dim='time')                   # liquid and ice water mass
        pr_lim, clwvi_lim = [1, 0.20]
        pr_th, clwvi_th = pr.quantile(pr_lim, dim=('lat', 'lon'), keep_attrs=True), clwvi.quantile(clwvi_lim, dim=('lat', 'lon'), keep_attrs=True) # remove large pr and small clwvi
        pr = pr.where((pr < pr_th) & (pr > 0), np.nan)     
        clwvi = clwvi.where(clwvi > clwvi_th, np.nan)     
        da = pr / clwvi
        # print(f'{var} lims: \n min: {da.min().data} \n max: {da.max().data}')
    else:
        da = mF.load_variable({var: True}, switch, dataset, experiment)
    return da


# ------------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #
def get_data(switch, var, dataset, experiment):
    da = get_variable(switch, var, dataset, experiment)
    da, vert_reg = pick_vert_reg(switch, da)
    da, hor_reg  = pick_hor_reg(switch, dataset, experiment, da)    # experiment needed as wap is loaded to pick region
    return da, f'{vert_reg}{hor_reg}'



# ------------------------
#    Calculate metrics
# ------------------------
@mF.timing_decorator()
def get_snapshot(da):
    plot = False
    if plot:
        import myFuncs_plots as mFd     
        for timestep in np.arange(0, len(da.time.data)):
            if 'plev' in da.dims:
                print('also has vertical dimensions')
            fig = mFd.plot_scene(da.isel(time=timestep), ax_title = timestep, vmin = -80, vmax = 80, cmap = 'RdBu')    #, vmin = 0, vmax = 60) #, cmap = 'RdBu')
            if mFd.show_plot(fig, show_type = 'cycle', cycle_time = 0.5):                             # 3.25 # show_type = [show, save_cwd, cycle] (cycle wont break the loop)
                break
    return da.isel(time=0)

@mF.timing_decorator()
def get_tMean(da):
    return da.mean(dim='time', keep_attrs=True)

@mF.timing_decorator()
def get_sMean(da):
    return da.weighted(np.cos(np.deg2rad(da.lat))).mean(dim=('lat','lon'), keep_attrs=True).compute() # dask objects require the compute part

# @mF.timing_decorator()
# def get_area(da, dim):
#     ''' Area covered in domain [% of domain]. Used for area of ascent and descent (wap) '''
#     mask = xr.where(da > 0, 1, 0)
#     area = ((mask*dim.aream).sum(dim=('lat','lon')) / np.sum(dim.aream)) * 100
#     return area



# ------------------------
#   Run / save metric
# ------------------------
# ------------------------------------------------------------------------------------ Get metric and metric name ----------------------------------------------------------------------------------------------------- #
def calc_metric(switchM, var_name, da, region):
    dim = mF.dims_class(da)
    for metric_name in [k for k, v in switchM.items() if v]:
        metric = None
        metric = get_snapshot(da)   if metric_name == 'snapshot'    else metric
        metric = get_tMean(da)      if metric_name == 'tMean'       else metric
        metric = get_sMean(da)      if metric_name == 'sMean'       else metric
        metric_name =f'{var_name}{region}_{metric_name}' 
        yield metric, metric_name


# ------------------------------------------------------------------------------------ Get dataset and save metric ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator(show_time = True)
def run_ls_metrics(switch_var, switchM, switch):
    print(f'variable: {mV.resolutions[0]} {mV.timescales[0]} data \n {[key for key, value in switch_var.items() if value]}')
    print(f'metric: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    for var_name in [k for k, v in switch_var.items() if v]:
        for dataset, experiment in mF.run_dataset(var_name):
            da, region = get_data(switch, var_name, dataset, experiment)
            for metric, metric_name in calc_metric(switchM, var_name, da, region):      
                # print(metric_name)      
                # print(metric)      
                path = mF.save_metric(switch, var_name, dataset, experiment, metric, metric_name)
                # print(path)

# ----------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {                                                                                              # Choose variable (can choose multiple)
        'pr':       False,  'clwvi':        False,   'pe':          False,                                      # Precipitation
        'wap':      False,                                                                                      # Circulation
        'hur':      True,  'hus':           False,                                                              # Humidity                             
        'tas':      False,  'ta':           False,  'stability':    False,                                      # Temperature
        'rlut':     False,  'rlds':         False,  'rlus':         False,  'netlw':    False,                  # Longwave radiation
        'rsut':     False,  'rsdt':         False,  'rsds':         False,  'rsus':     False, 'netsw': False,  # Shortwave radiation
        'lcf':      False,  'hcf':          False,                                                              # Cloud fraction
        'zg':       False,                                                                                      # Geopotential height
        'hfss':     False,  'hfls':         False,                                                              # Surface fluxes
        'h':        False,  'h_anom2':      False,                                                              # Moist Static Energy
        }
    
    switchM = {                                                                         # choose metric type (can choose multiple)
        'snapshot': True,   'tMean':    True,   'sMean':    True,                       # type 
        }

    switch = {                                                                                                  # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                              # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           True,  '250hpa':       False,  'vMean':    False,  # mask data: vertical (3D variables are: wap, hur, ta, zg, hus)
        'ascent_fixed':         False,  'descent_fixed':    False,  'ocean':        False,                      # mask data: horizontal 
        'ascent':               False,  'descent':          False,                                              # mask data: horizontal (can apply both ocean and ascent/descent together)
        'save_folder_desktop':  True,   'save_scratch':     False,  'save':         False                       # Save
        }
    
    run_ls_metrics(switch_var, switchM, switch)

















