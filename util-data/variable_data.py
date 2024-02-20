'''
# ------------------------
#   Getting variable
# ------------------------
These variables are calculated from base variables (the base variables are at times saved in scratch for quick access) (the calculated variables are rarely stored)
Some variables need to be calculated, and sometimes a horizontal or vertical region is picked out

To use:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_data as vD

da = vD.get_variable_data(switch, var_name, dataset, experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import re


# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_base as vB
import get_data.dimensions_data as dD
import get_data.pe_var as pE


sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc      as mean_calc



# ------------------------
#      Pick region
# ------------------------
# ------------------------------------------------------------------------------------- Vertical mask --------------------------------------------------------------------------------------------------- #
def pick_vMean(da, plevs0 = 850e2, plevs1 = 0):         
    ''' # free troposphere (as most values at 1000 hPa and 925 hPa over land are NaN)
        # Where there are no values, exclude the associated pressure levels from the weights '''
    da = da.sel(plev = slice(plevs0, plevs1))
    w = ~np.isnan(da) * da['plev']                      
    da = (da * w).sum(dim='plev') / w.sum(dim='plev') 
    return da

def pick_vert_reg(switch, da):
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        number = re.findall(r'\d+', met_type)
        if number and 'hpa' in met_type: # if there is a number and hpa in string
            level = int(re.findall(r'\d+', met_type)[0]) * 10**2
            da, region = [da.sel(plev = level), f'_{number}hpa']
        if met_type == 'vMean':
            da, region = [pick_vMean(da), 'vMean']
    return da, region


# ------------------------------------------------------------------------------------ Horizontal mask --------------------------------------------------------------------------------------------------- #
def pick_hor_reg(switch, dataset, experiment, da, resolution, timescale):
    ''' Ascent/descent region based on 500 hPa vertical pressure velocity (wap)
    # loading data deals with picking out ocean (as it can be done before or after interpolation) '''
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        if ('descent' in switch and met_type == 'descent') or \
            ('ascent' in switch and met_type == 'ascent'):
            wap500 = vB.load_variable(switch, 'wap', dataset, experiment, resolution, timescale).sel(plev = 500e2)         
            da, region = [da.where(wap500 > 0), '_d']  if met_type == 'descent' else [da, region]
            da, region = [da.where(wap500 < 0), '_a']  if met_type == 'ascent'  else [da, region]
        if ('descent_fixed' in switch and met_type == 'descent_fixed') or \
            ('ascent_fixed' in switch and met_type == 'ascent_fixed'):
            wap500 = vB.load_variable(switch, 'wap', dataset, experiment, resolution, timescale).sel(plev = 500e2).mean(dim='time')
            da, region = [da.where(wap500 > 0), '_fd']  if met_type == 'descent_fixed' else [da, region]
            da, region = [da.where(wap500 < 0), '_fa']  if met_type == 'ascent_fixed'  else [da, region]
        region = f'_o{region}'  if met_type == 'ocean'         else region                    
    return da, region



# ------------------------
#  Calculated variables
# ------------------------
# ------------------------------------------------------------------------------------ calculate --------------------------------------------------------------------------------------------------- #
def get_clouds(switch, var_name, dataset, experiment, resolution, timescale):
    ''' # can also do 250 up (in schiro 'spread paper') '''
    da = vB.load_variable({'cl':True}, switch, dataset, experiment, resolution, timescale)
    da = da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev') if var_name == 'lcf' else da
    da = da.sel(plev = slice(400e2, 0)).max(dim = 'plev')      if var_name == 'hcf' else da  
    return da

def get_stability(switch, var_name, dataset, experiment, resolution, timescale):
    ''' # Differnece in potential temperature between two vertical sections 
    # Temperature at pressure levels (K) 
    # Where there are no temperature values, exclude the associated pressure levels from the weights'''
    da = vB.load_variable({'ta': True}, switch, dataset, experiment, resolution, timescale)                    
    theta =  da * (1000e2 / da['plev'])**(287/1005) 
    plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
    da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
    w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']                 
    da = ((da1 * w1).sum(dim='plev') / w1.sum(dim='plev')) - ((da2 * w2).sum(dim='plev') / w2.sum(dim='plev'))
    return da

def get_netlw(switch, var_name, dataset, experiment, resolution, timescale):
    rlds, rlus, rlut = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['rlds', 'rlus', 'rlut']]
    da = -rlds + rlus - rlut
    return da

def get_netsw(switch, var_name, dataset, experiment, resolution, timescale):
    rsdt, rsds, rsus, rsut = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['rsdt', 'rsds', 'rsus', 'rsut']]
    da = rsdt - rsds + rsus - rsut
    return da

def get_mse(switch, var_name, dataset, experiment, resolution, timescale):
    '''  # h - Moist Static Energy (MSE) '''
    c_p, L_v = dD.dims_class.c_p, dD.dims_class.L_v
    ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['ta', 'zg', 'hus']]
    da = c_p * ta + zg + L_v * hus
    return da

def get_mse_anom2(switch, var_name, dataset, experiment, resolution, timescale):
    '''# MSE variance from the tropical mean  '''
    c_p, L_v = dD.dims_class.c_p, dD.dims_class.L_v
    ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['ta', 'zg', 'hus']]
    da = c_p * ta + zg + L_v * hus
    da, _ = pick_vert_reg(switch, dataset, da)
    da_sMean = mean_calc.get_sMean(da)
    da_anom = da - da_sMean
    da = da_anom**2
    return da

def get_pe(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Precipitation efficiency 
    pr is # mm/m^2/day
    # liquid and ice water mass
    # remove small clwvi '''
    pr = vB.load_variable({'pr': True}, switch, dataset, experiment, resolution, timescale = 'daily').resample(time='1MS').mean(dim='time')    
    clwvi = vB.load_variable({'clwvi': True}, switch, dataset, experiment, resolution, timescale).resample(time='1MS').mean(dim='time')                   
    pr_lim, clwvi_lim = [1, 0.20]
    pr_th, clwvi_th = pr.quantile(pr_lim, dim=('lat', 'lon'), keep_attrs=True), clwvi.quantile(clwvi_lim, dim=('lat', 'lon'), keep_attrs=True) 
    pr = pr.where((pr < pr_th) & (pr > 0), np.nan)     
    clwvi = clwvi.where(clwvi > clwvi_th, np.nan)     
    da = pr / clwvi
    return da

# ------------------------------------------------------------------------------------ summarize --------------------------------------------------------------------------------------------------- #
def calc_variable(switch, var_name, dataset, experiment, resolution, timescale):
    ''' Gets variable (some need to be calculated) '''
    # print(f'getting {var_name}{[key for key, value in switch.items() if value]}')    
    if var_name in ['lcf', 'hcf', 'stability', 'netlw', 'netsw', 'h', 'h_anom2']:
        da = get_clouds(switch, var_name, dataset, experiment, resolution, timescale)          if var_name in ['lcf', 'hcf']   else None
        da = get_stability(switch, var_name, dataset, experiment, resolution, timescale)       if var_name == 'stability'      else da
        da = get_netlw(switch, var_name, dataset, experiment, resolution, timescale)           if var_name == 'netlw'          else da
        da = get_netsw(switch, var_name, dataset, experiment, resolution, timescale)           if var_name == 'netsw'          else da
        da = get_netsw(switch, var_name, dataset, experiment, resolution, timescale)           if var_name == 'netsw'          else da
        da = get_mse(switch, var_name, dataset, experiment, resolution, timescale)             if var_name == 'h'              else da
        da = get_mse_anom2(switch, var_name, dataset, experiment, resolution, timescale)       if var_name == 'h_anom2'        else da
        da = pE.get_pe(switch, var_name, dataset, experiment, resolution, timescale)              if var_name == 'pe'             else da
    else:
        da = vB.load_variable({var_name: True}, switch, dataset, experiment, resolution, timescale)    # basic metrics
    return da

def get_variable_data(switch, var_name, dataset, experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0]):
    ''' Picks region of variable '''
    da = calc_variable(switch, var_name, dataset, experiment, resolution, timescale)
    da, vert_reg = pick_vert_reg(switch, da) if 'plev' in da.dims else [da, '']
    da, hor_reg  = pick_hor_reg(switch, dataset, experiment, da, resolution, timescale)    # experiment needed as wap is loaded to pick region
    return da, f'{vert_reg}{hor_reg}'



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':

    switch_var = {                                                  # Choose variable
        'pr':           False,  'pe':           False,              # Precipitation (pr not calculated, used for testing that it can call variable_base)               
        'wap':          False,                                      # Circulation
        'hur':          False,  'hus' :         False,              # Humidity               
        'stability':    False,                                      # Temperature
        'netlw':        False,                                      # Longwave radiation
        'netsw':        False,                                      # Shortwave radiation
        'lcf':          False,  'hcf':          False,              # Cloud fraction
        'zg':           False,                                      # Height coordinates
        'h':            True,  'h_anom2':      False,              # Moist Static Energy
        }

    switch = {                                                                                                 # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                             # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           True,   '250hpa':  False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                         # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':  False,  'descent':  False,      # horizontal mask
        }

    var_name = next((key for key, value in switch_var.items() if value), None)
    da, region = get_variable_data(switch = switch, var_name = var_name, dataset = cD.datasets[0], experiment = cD.experiments[0])
    print(da)





