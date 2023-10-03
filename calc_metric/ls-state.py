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
        da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/{var}/{source}/{dataset}_{var}_{mV.timescales[0]}_{experiment}_{mV.resolutions[0]}.nc')[f'{var}']
    if switch['gadi_data']:
        if var == 'hur':                                                  # Bolton's formula (calculated from specific humidity and temperature)
            r = gD.get_var_data(source, dataset, experiment, 'hus', switch)       # unitless (kg/kg)
            T = gD.get_var_data(source, dataset, experiment, 'ta', switch)        # degrees Kelvin
            p = T['plev']                                                 # Pa
            e_s = 611.2 * np.exp(17.67*(T-273.15)/(T-29.66))              # saturation water vapor pressure
            r_s = 0.622 * e_s/p
            da = (r/r_s)*100                                              # relative humidity could do e/e_s
        elif var == 'stability':                                            # Calculated as differnece in potential temperature at two height slices
            da = gD.get_var_data(source, dataset, experiment, 'ta', switch)
            theta =  da * (1000e2 / da.plev)**(287/1005)                  # theta = T (P_0/P)^(R_d/C_p)
            plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]               # pressure levels in ERA are reversed to cmip
            da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))] if not dataset == 'ERA5' else [theta.sel(plev=slice(plevs1[1], plevs1[0])), theta.sel(plev=slice(plevs2[1], plevs2[0]))] 
            da = ((da1 * da1.plev).sum(dim='plev') / da1.plev.sum(dim='plev')) - ((da2 * da2.plev).sum(dim='plev') / da2.plev.sum(dim='plev'))   

        elif var in ['lcf', 'hcf']:
            p_hybridsigma = gD.get_var_data(source, dataset, experiment, 'p_hybridsigma', switch)
            da = gD.get_var_data(source, dataset, experiment, 'cl', switch)
            plevs1, plevs2 = [250e2, 0], [1500e2, 600e2]
            da = da.where((p_hybridsigma <= plevs1[0]) & (p_hybridsigma >= plevs1[1]), 0).max(dim='lev') if switch['hcf'] else da
            da = da.where((p_hybridsigma <= plevs2[0]) & (p_hybridsigma >= plevs2[1]), 0).max(dim='lev') if switch['lcf'] else da
        else:
            da = gD.get_var_data(source, dataset, experiment, f'{var}', switch)
    return da


# ----------------------------------------------------------------------------------------- pick region ----------------------------------------------------------------------------------------------------- #
def pick_vert_reg(switch, dataset, da):
    region = ''
    if switch['250hpa']:
        da = da.sel(plev = 250e2)
        region = '_250hpa'
    if switch['500hpa']:
        da = da.sel(plev = 500e2)
        region = '_500hpa'
    if switch['700hpa']:
        da = da.sel(plev = 700e2)
        region = '_700hpa'
    if switch['vMean']:
        plevs = [850e2, 0]
        plevs = slice(plevs[0], plevs[1]) if not dataset == 'ERA5' else slice(plevs[1], plevs[0])
        da = da.sel(plev=plevs)
        da = (da * da.plev).sum(dim='plev') / da.plev.sum(dim='plev') # free troposphere (most values at 1000 hPa over land are NaN)
        region = ''
    return da, region

def pick_hor_reg(switch, source, dataset, experiment, da):
    ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
    region = ''
    if switch['descent']:
        wap500 = load_data(switch, source, dataset, experiment, 'wap').sel(plev = 500e2)
        da = da.where(wap500>0)
        region = '_d'
    if switch['ascent']:
        wap500 = load_data(switch, source, dataset, experiment, 'wap').sel(plev = 500e2)
        da = da.where(wap500<0)
        region = '_a'
    if switch['ascent']:
        region = '_ocean'
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
    return da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True)

@mF.timing_decorator
def calc_area(da):
    ''' Area covered in domain [% of domain]'''
    dim = mC.dims_class(da)
    da = xr.where(da>0, 1, 0)
    area = np.sum(da * np.transpose(dim.aream3d, (2, 0, 1)), axis=(1,2)) / np.sum(dim.aream)
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
            print(f'{var}')
            if not mV.data_available(source, dataset, experiment, var):
                continue
            da =           load_data(switch, source, dataset, experiment, var)
            da, vert_reg = pick_vert_reg(switch, dataset, da) if var in ['hur', 'wap'] else [da, '']
            da, hor_reg =  pick_hor_reg(switch, source, dataset, experiment, da)
            run_metric(switchM, switch, source, dataset, experiment, var, da, vert_reg, hor_reg)

def run_experiment(switch_var, switchM, switch, source, dataset):
    for experiment in mV.experiments:
        if not mV.data_available(source, dataset, experiment, var = '', switch = switch):
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

    switch_var = {
        # choose variable (can choose multiple)
        'hur':                False,
        'rlut':               False, 
        'tas':                False,
        'wap':                False,
        'stability':          True,
        'lcf':                False,
        'hcf':                False,
        }
    
    switchM = {
        # choose type of metric (can choose multiple)
        'snapshot':           True, 
        'tMean':              True, 
        'sMean':              True, 
        'area':               False,
        }

    switch = {
        # choose type of data to calculate metric on
        'constructed_fields': False, 
        'sample_data':        False,
        'gadi_data':          True,

        # Choose vertical region (only affects wap, hur)
        '250hpa':             False,
        '500hpa':             False,
        '700hpa':             False,
        'vMean':              False,

        # Choose horizontal region
        'ascent':             False,
        'descent':            False,
        'ocean_mask':         True,
        
        # save
        'save':               True,
        'save_to_desktop':    False
        }
    
    run_large_scale_state_metrics(switch_var, switchM, switch)
    



