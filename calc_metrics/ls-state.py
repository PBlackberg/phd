import numpy as np
import xarray as xr
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import constructed_fields as cF
import get_data as gD
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myClasses as mC
import myFuncs as mF



# ------------------------
#       Get data
# ------------------------
# ---------------------------------------------------------------------------------------- load data ----------------------------------------------------------------------------------------------------- #
@mF.timing_decorator
def load_data(switch, source, dataset, experiment, var):
    if switch['constructed_fields']:
        da = cF.var3D
    if switch['sample_data']:
        if var in ['lcf', 'hcf']:
            da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/cl/{source}/{dataset}_cl_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')['cl']
            da = da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev') if var == 'lcf' else da
            da = da.sel(plev = slice(400e2, 0)).max(dim = 'plev')      if var == 'hcf' else da # can also do 250 up (in schiro spread paper)
        else:
            da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/{var}/{source}/{dataset}_{var}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')[f'{var}']

    if switch['gadi_data']:
        if var == 'hur_calc':                                                         # Relative humidity (calculated from tempearture and specific humidity)
                q = gD.get_var_data(source, dataset, experiment, 'hus', switch)       # unitless (kg/kg)
                T = gD.get_var_data(source, dataset, experiment, 'ta', switch)        # degrees Kelvin
                p = T['plev']                                                         # Pa 
                r = q / (1 - q)                                                       # q = r / (1+r)
                e_s = 611.2 * np.exp(17.67*(T-273.15)/(T-29.66))                      # saturation water vapor pressure (also: e_s = 2.53*10^11 * np.exp(-B/T) (both from lecture notes), the Goff-Gratch Equation:  10^(10.79574 - (1731.464 / (T + 233.426)))
                r_s = 0.622 * e_s/(p-e_s)                                             # from book
                da = (r/r_s)*((1+(r_s/0.622)) / (1+(r/0.622)))*100                    # relative humidity (from book)

        elif var == 'stability':                                                        # Calculated as differnece in potential temperature between two height sections
            da = gD.get_var_data(source, dataset, experiment, 'ta', switch)             # Temperature at pressure levels (K)
            theta =  da * (1000e2 / da['plev'])**(287/1005) 
            plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
            da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
            w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']         # Where there are no temperature values, exclude the associated pressure levels from the weights
            da = ((da1 * w1).sum(dim='plev') / w1.sum(dim='plev')) - ((da2 * w2).sum(dim='plev') / w2.sum(dim='plev'))

        elif var == 'netlw':    
            rlds = gD.get_var_data(source, dataset, experiment, 'rlds', switch) 
            rlus = gD.get_var_data(source, dataset, experiment, 'rlus', switch) 
            rlut = gD.get_var_data(source, dataset, experiment, 'rlut', switch) 
            da = -rlds + rlus - rlut

        elif var == 'netsw':   
            rsdt = gD.get_var_data(source, dataset, experiment, 'rsdt', switch) 
            rsds = gD.get_var_data(source, dataset, experiment, 'rsds', switch) 
            rsus = gD.get_var_data(source, dataset, experiment, 'rsus', switch) 
            rsut = gD.get_var_data(source, dataset, experiment, 'rsut', switch) 
            da = rsdt - rsds + rsus - rsut

        elif var == 'mse':   
            c_p, L_v = 1.005, 2.256e6
            ta = gD.get_var_data(source, dataset, experiment, 'ta', switch) 
            zg = gD.get_var_data(source, dataset, experiment, 'zg', switch) 
            hus = gD.get_var_data(source, dataset, experiment,'hus', switch) 
            da = c_p*ta + zg + L_v*hus
        else:
            da = gD.get_var_data(source, dataset, experiment, var, switch)
    return da


# ----------------------------------------------------------------------------------------- pick region ----------------------------------------------------------------------------------------------------- #
def pick_vert_reg(switch, dataset, da):
    region = ''
    if switch['250hpa']:
        region = '_250hpa'
        da = da.sel(plev = 250e2)
    if switch['500hpa']:
        region = '_500hpa'
        da = da.sel(plev = 500e2)
    if switch['700hpa']:
        region = '_700hpa'
        da = da.sel(plev = 700e2)
    if switch['vMean']:
        region = ''
        plevs = [850e2, 0]                                # free troposphere (as most values at 1000 hPa  and 925 hPa over land are NaN)
        plevs = slice(plevs[0], plevs[1]) if not dataset == 'ERA5' else slice(plevs[1], plevs[0])
        da = da.sel(plev=plevs)
        w = ~np.isnan(da) * da['plev']                    # Where there are no values, exclude the associated pressure levels from the weights
        da = (da * w).sum(dim='plev') / w.sum(dim='plev') 
    return da, region

def pick_hor_reg(switch, source, dataset, experiment, da):
    ''' Ascent/descent region based on 500 hPa vertical pressure velocity (wap)'''
    region = ''
    if switch['descent'] or switch['ascent']:
        wap500 = load_data(switch, source, dataset, experiment, 'wap').sel(plev = 500e2)
        region = '_d'           if switch['descent'] else '_a'
        da = da.where(wap500>0) if switch['descent'] else da.where(wap500<0)

    if switch['ocean']:
        region = f'_o{region}' # gD.get_var_data() deals with picking out the ocean
    return da, region



# ------------------------
#    Calculate metrics
# ------------------------
@mF.timing_decorator
def get_snapshot(da):
    snapshot = da.isel(time=0)
    return snapshot

@mF.timing_decorator
def get_tMean(da):
    tMean = da.mean(dim='time', keep_attrs=True)
    return tMean 

@mF.timing_decorator
def calc_sMean(da):
    aWeights = np.cos(np.deg2rad(da.lat))
    sMean = da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True).compute() # dask objects require the compute part
    return sMean

@mF.timing_decorator
def calc_area(da):
    ''' Area covered in domain [% of domain] '''
    dim = mC.dims_class(da)
    mask = xr.where(da>0, 1, 0)
    area = ((mask*dim.aream).sum(dim=('lat','lon')) / dim.aream.sum(dim=('lat','lon'))) * 100
    return area



# ------------------------
#   Run / save metric
# ------------------------
# -------------------------------------------------------------------------------------------- Get metric and save ----------------------------------------------------------------------------------------------------- #
def get_metric(switch, source, dataset, experiment, var, da, vert_reg, hor_reg, metric):
    da_calc, metric_name = None, None
    if metric == 'snapshot':
        metric_name =f'{var}{vert_reg}{hor_reg}_snapshot' 
        da_calc = get_snapshot(da)

    if metric == 'tMean':
        metric_name =f'{var}{vert_reg}{hor_reg}_tMean'
        da_calc = get_tMean(da)

    if metric == 'sMean':
        metric_name =f'{var}{vert_reg}{hor_reg}_sMean' 
        da_calc = calc_sMean(da)

    if metric == 'area':
        metric_name =f'{var}{vert_reg}{hor_reg}_area'
        da_calc = calc_area(da)

    mF.save_in_structured_folders(da_calc, f'{mV.folder_save[0]}/metrics', var, metric_name, source, dataset, mV.timescales[0], experiment, mV.resolutions[0])                       if switch['save'] else None
    mF.save_file(xr.Dataset(data_vars = {metric_name: da_calc}), f'{home}/Desktop/{metric_name}', f'{dataset}_{metric_name}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc') if switch['save_to_desktop'] else None


# ------------------------------------------------------------------------------------------------- run metric ----------------------------------------------------------------------------------------------------- #
def run_metric(switchM, switch, source, dataset, experiment, var, da, vert_reg, hor_region):
    for metric in [k for k, v in switchM.items() if v]:
            get_metric(switch, source, dataset, experiment, var, da, vert_reg, hor_region, metric)

def run_variable(switch_var, switchM, switch, source, dataset, experiment):
    for var in [k for k, v in switch_var.items() if v]:
            print(f'\t\t\t{var}')
            if not mV.data_available(source, dataset, experiment, var):                     # skips calc for models that do not have variable
                continue
            da =           load_data(switch, source, dataset, experiment, var)
            da, vert_reg = pick_vert_reg(switch, dataset, da) if var in ['hur', 'wap', 'mse'] else [da, ''] 
            da, hor_reg  = pick_hor_reg(switch, source, dataset, experiment, da)
            run_metric(switchM, switch, source, dataset, experiment, var, da, vert_reg, hor_reg)

def run_experiment(switch_var, switchM, switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment, var = '', switch = switch):   # skips invalid experiment combinations (like obs, or cmip5 model with ssp585)
            continue
        print(f'\t\t {experiment}') if experiment else print(f'\t observational dataset')
        run_variable(switch_var, switchM, switch, source, dataset, experiment)

def run_dataset(switch_var, switchM, switch):
    for dataset in mV.datasets:
        source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
        print(f'\t{dataset} ({source})')
        run_experiment(switch_var, switchM, switch, source, dataset)

@mF.timing_decorator
def run_large_scale_state_metrics(switch_var, switchM, switch):
    print(f'Running {mV.resolutions[0]} {mV.timescales[0]} data')
    print(f'metric: {[key for key, value in switch_var.items() if value]}')
    print(f'metric_type: {[key for key, value in switchM.items() if value]}')
    print(f'settings: {[key for key, value in switch.items() if value]}')
    run_dataset(switch_var, switchM, switch)


# ------------------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    switch_var = {                                                                                   # choose variable (can choose multiple)
        'wap':   False,                                                                              # circulation
        'hur':   False, 'hur_calc':  False, 'hus':       False,                                      # humidity
        'tas':   False, 'stability': False,                                                          # temperature
        'netlw': False, 'rlds':      False, 'rlus': False, 'rlut': False,                            # longwave radiation
        'netsw': False, 'rsdt':      False, 'rsds': False, 'rsus': False, 'rsut': False,             # shortwave radiation
        'lcf':   False, 'hcf':       False,                                                          # cloud fraction
        'mse':   False, 'zg':        False, 'hfss': False, 'hfls': False,                            # moist static energy
        }
    
    switchM = {                                                                                      # choose metric type (can choose multiple)
        'snapshot': True, 'tMean': True, 'sMean': True, 'area': False,                               # type 
        }

    switch = {                                                                                       # choose data to use and mask
        'constructed_fields': False, 'sample_data': True, 'gadi_data': False,                        # data to use
        '250hpa':             False, '500hpa':      False, '700hpa':   False, 'vMean': False,        # mask: vertical (only affects wap, hur)
        'ascent':             False, 'descent':     False, 'ocean':    False,                        # mask: horizontal (can apply both ocean and ascent/descent together)
        'save_to_desktop':    False, 'save':        True,                                            # save
        }
    run_large_scale_state_metrics(switch_var, switchM, switch)

