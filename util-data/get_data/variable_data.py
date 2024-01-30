'''
# ------------------------
#   Getting variable
# ------------------------
Some variables need to be calculated, and sometimes a horizontal or vertical region is picked out
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np



# ------------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-data')
import get_data.variable_base as vB

sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc      as mean_calc

sys.path.insert(0, f'{os.getcwd()}/util-core')
import myFuncs          as mF     



# ------------------------
#      Pick region
# ------------------------
# ------------------------------------------------------------------------------------- Vertical mask --------------------------------------------------------------------------------------------------- #
def pick_vMean(da, plevs0 = 850e2, plevs1 = 0):         # free troposphere (as most values at 1000 hPa and 925 hPa over land are NaN)                                            #  
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
        da, region = [pick_vMean(da), 'vMean']          if met_type == 'vMean'  and 'plev' in da.dims   else [da, region]
    return da, region


# ------------------------------------------------------------------------------------ Horizontal mask --------------------------------------------------------------------------------------------------- #
def pick_hor_reg(switch, dataset, experiment, da):
    ''' Ascent/descent region based on 500 hPa vertical pressure velocity (wap)'''
    da, region = da, ''
    for met_type in [k for k, v in switch.items() if v]:
        wap500 = vB.load_variable(switch, 'wap', dataset, experiment).sel(plev = 500e2) if ('descent' in switch and met_type == 'descent') or ('ascent' in switch and met_type == 'ascent') else None
        da, region = [da.where(wap500 > 0), '_d']  if met_type == 'descent' else [da, region]
        da, region = [da.where(wap500 < 0), '_a']  if met_type == 'ascent'  else [da, region]

        wap500 = vB.load_variable(switch, 'wap', dataset, experiment).sel(plev = 500e2).mean(dim='time') if ('descent_fixed' in switch and met_type == 'descent_fixed') or ('ascent_fixed' in switch and met_type == 'ascent_fixed') else None
        da, region = [da.where(wap500 > 0), '_fd']  if met_type == 'descent_fixed' else [da, region]
        da, region = [da.where(wap500 < 0), '_fa']  if met_type == 'ascent_fixed'  else [da, region]
        region = f'_o{region}'                      if met_type == 'ocean' else region   # loading data deals with picking out ocean (as it can be done before or after interpolation)                         
    return da, region



# ------------------------
#   Calculate variable
# ------------------------
# --------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
def get_variable(switch, var_name, dataset, experiment):
    # print(f'getting {var_name}{[key for key, value in switch.items() if value]}')
    if var_name in ['lcf', 'hcf']:
        da = vB.load_variable({'cl':True}, switch, dataset, experiment)
        da = da.sel(plev = slice(1000e2, 600e2)).max(dim = 'plev') if var_name == 'lcf' else da
        da = da.sel(plev = slice(400e2, 0)).max(dim = 'plev')      if var_name == 'hcf' else da  # can also do 250 up (in schiro 'spread paper')
    elif var_name == 'stability':                                                                # Differnece in potential temperature between two vertical sections
        da = vB.load_variable({'ta': True}, switch, dataset, experiment)                    # Temperature at pressure levels (K)
        theta =  da * (1000e2 / da['plev'])**(287/1005) 
        plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
        da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
        w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']                 # Where there are no temperature values, exclude the associated pressure levels from the weights
        da = ((da1 * w1).sum(dim='plev') / w1.sum(dim='plev')) - ((da2 * w2).sum(dim='plev') / w2.sum(dim='plev'))
    elif var_name == 'netlw':    
        rlds, rlus, rlut = [vB.load_variable({var: True}, switch, dataset, experiment) for var in ['rlds', 'rlus', 'rlut']]
        da = -rlds + rlus - rlut
    elif var_name == 'netsw':           
        rsdt, rsds, rsus, rsut = [vB.load_variable({var: True}, switch, dataset, experiment) for var in ['rsdt', 'rsds', 'rsus', 'rsut']]
        da = rsdt - rsds + rsus - rsut
    elif var_name == 'h':                                                                        # h - Moist Static Energy (MSE)
        c_p, L_v = mF.dims_class.c_p, mF.dims_class.L_v
        ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment) for var in ['ta', 'zg', 'hus']]
        da = c_p * ta + zg + L_v * hus
    elif var_name == 'h_anom2':                                                                  # MSE variance from the tropical mean
        c_p, L_v = mF.dims_class.c_p, mF.dims_class.L_v
        ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment) for var in ['ta', 'zg', 'hus']]
        da = c_p * ta + zg + L_v * hus
        da, _ = pick_vert_reg(switch, dataset, da)
        da_sMean = mean_calc.get_sMean(da)
        da_anom = da - da_sMean
        da = da_anom**2
        # plot_object = da.isel(time=0).plot()
        # fig = plot_object.figure
        # fig.savefig(f'{os.getcwd()}/test/plot_test/test2.png')
    elif var_name == 'pe':
        pr = mF.load_variable({'pr': True}, switch, dataset, experiment, timescale = 'daily').resample(time='1MS').mean(dim='time')    # mm/m^2/day
        clwvi = mF.load_variable({'clwvi': True}, switch, dataset, experiment).resample(time='1MS').mean(dim='time')                   # liquid and ice water mass
        pr_lim, clwvi_lim = [1, 0.20]
        pr_th, clwvi_th = pr.quantile(pr_lim, dim=('lat', 'lon'), keep_attrs=True), clwvi.quantile(clwvi_lim, dim=('lat', 'lon'), keep_attrs=True) # remove large pr and small clwvi
        pr = pr.where((pr < pr_th) & (pr > 0), np.nan)     
        clwvi = clwvi.where(clwvi > clwvi_th, np.nan)     
        da = pr / clwvi
        # print(f'{var} lims: \n min: {da.min().data} \n max: {da.max().data}')
    else:
        da = vB.load_variable({var_name: True}, switch, dataset, experiment)
    return da


# ------------------------------------------------------------------------------------------- Get --------------------------------------------------------------------------------------------------- #
def get_variable_data(switch, var_name, dataset, experiment):
    da = get_variable(switch, var_name, dataset, experiment)
    da, vert_reg = pick_vert_reg(switch, da)
    da, hor_reg  = pick_hor_reg(switch, dataset, experiment, da)    # experiment needed as wap is loaded to pick region
    return da, f'{vert_reg}{hor_reg}'



# ------------------------
#         Test
# ------------------------
# ------------------------------------------------------------------------------------- Choose what to run ----------------------------------------------------------------------------------------------------- #
if __name__ == '__main__':
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import myVars          as mV
    import myFuncs_plots    as mFp     

    switch_var = {                                                  # Choose variable
        'pr':           False,  'pe':           False,              # Precipitation (pr not calculated, for test, this script can also get base variables)               
        'wap':          True,                                      # Circulation
        'hur':          False,  'hus' :         False,              # Humidity               
        'stability':    False,                                      # Temperature
        'netlw':        False,                                      # Longwave radiation
        'netsw':        False,                                      # Shortwave radiation
        'lcf':          False,  'hcf':          False,              # Cloud fraction
        'zg':           False,                                      # Height coordinates
        'h':            False,  'h_anom2':      False,              # Moist Static Energy
        }

    switch = {                                                                                                 # choose data to use and mask
        'constructed_fields':   False,  'test_sample':      False,                                             # data to use (test_sample uses first file (usually first year))
        '700hpa':               False,  '500hpa':           True,   '250hpa':  False,  'vMean':    False,      # vertical mask (3D variables are: wap, hur, ta, zg, hus)
        'ocean_mask':           False,                                                                         # horizontal mask
        'ascent_fixed':         False,  'descent_fixed':    False,  'ascent':  False,  'descent':  False,      # horizontal mask
        }

    var_name = next((key for key, value in switch_var.items() if value), None)
    da, region = get_variable_data(switch = switch, var_name = var_name, dataset = mV.datasets[0], experiment = mV.experiments[0])
    print(da)
    # mFp.get_snapshot(da, plot = True, show_type = 'cycle')  # show_type = [show, save_cwd, cycle] 





