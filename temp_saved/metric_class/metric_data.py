'''
# ------------------------
#    Metric data
# ------------------------
This script creates a python object with features of a metric like [variable_name, metric_name, colormap, plot_label, plot_color]
The script also has a function for gettin the metric from where it is saved
'''



# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import xarray as xr


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util-core')
import myFuncs as mF
import myVars as mV


# ------------------------
#   Define metric object
# ------------------------
# --------------------------------------------------------------------------------------- Define class ----------------------------------------------------------------------------------------------------- #
class metric_class():
    ''' Gives metric: name (of saved dataset), option (data array in dataset), label, cmap, color (used for plots of calculated metrics) '''
    def __init__(self, var, met_name, cmap, label, color='k'):
        self.var           = var
        self.name          = met_name
        self.label         = label
        self.cmap          = cmap
        self.color         = color



# -----------------------
#  Get metric object
# -----------------------
# ------------------------------------------------------------------------------------ add variation of metric ----------------------------------------------------------------------------------------------------- #
def v_reg(switch = {'a':False}):
    ''' Vertical region of 3d variable that metric is calculated from '''
    region = ''
    for met_type in [k for k, v in switch.items() if v]:
        region = '_250hpa' if met_type in ['250hpa'] else region
        region = '_700hpa' if met_type in ['700hpa'] else region
        region = '_500hpa' if met_type in ['500hpa'] else region
    return region

def h_reg(switch = {'a':False}):
    ''' Horizontal region of variable that metric is calculated from '''
    region = ''
    for met_type in [k for k, v in switch.items() if v]:
        region = '_d'           if met_type in ['descent']          else region
        region = '_a'           if met_type in ['ascent']           else region
        region = '_fd'          if met_type in ['descent_fixed']    else region
        region = '_fa'          if met_type in ['ascent_fixed']     else region
        region = f'_o{region}'  if met_type in ['ocean']            else region
    return region

def m_type(switch = {'snapshot':True}):
    ''' Metric type (mostly for ls-state metrics)'''
    metric_type = ''
    for met_type in [k for k, v in switch.items() if v]:
        metric_type = '_snapshot' if met_type in ['snapshot']   else metric_type
        metric_type = '_sMean'    if met_type in ['sMean']      else metric_type
        metric_type = '_tMean'    if met_type in ['tMean']      else metric_type
        metric_type = '_area_pos' if met_type in ['area_pos']   else metric_type
        metric_type = ''          if met_type in ['other']      else metric_type
    return metric_type

def c_prctile(prctile = '95'):
    ''' Convective percentile (this is for convective object oriented metrics) '''
    if prctile:
        prctile = f'_{prctile}thprctile'
    return prctile

def t_type(switch = {'fixed area':False}):
    ''' Convective threshold type (this is for convective object oriented metrics) '''
    threshold = ''
    for met_type in [k for k, v in switch.items() if v]:
        threshold = '_fixed_area' if met_type in ['fixed area'] else threshold
    return threshold

def cmap_label_exceptions(metric, switch, cmap, label):
    ''' Exeptions of features: Change cmap in a few cases
    RbBu        - Change with warming 
    Blue / Red  - wap_reg '''
    for met_type in [k for k, v in switch.items() if v]:
        cmap = 'RdBu_r' if met_type in ['change with warming'] else cmap
        cmap = 'Reds'   if met_type in ['descent'] and metric in ['wap'] else cmap   # wap normally has positive and negative values, but not in descent / ascent
        cmap = 'Blues'  if met_type in ['ascent']  and metric in ['wap'] else cmap

    for met_type in [k for k, v in switch.items() if v]:                            # add per K on change with warming plot labels
        label = '{}{}{}'.format(label[:-1], r'K$^{-1}$', label[-1:]) if met_type in ['per kelvin', 'per ecs'] and not label == 'ECS [K]' else label 
    return cmap, label


# ------------------------------------------------------------------ create metric object ----------------------------------------------------------------------------------------------------- #
def get_metric_class(metric = 'wap', switchM = {'500hpa': True, 'descent_fixed': True, 'tMean': True}, prctile = '95'):
    ''' Used for loading and plotting metrics 
    The sections refer to metric scripts   
    '''
    var, name, label, cmap, color = [None, None, 'test', 'Greys', 'k']

# ------------------------------------------------------------------------ Pr_metrics ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['pr',        f'pr{m_type(switchM)}',                                        r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr']           else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['pr',        f'pr_90{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_90']        else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['pr',        f'pr_95{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_95']        else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['pr',        f'pr_97{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_97']        else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_99{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_99']        else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_rx1day{m_type(switchM)}',                                 r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_rx1day']    else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_rx5day{m_type(switchM)}',                                 r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_rx5day']    else [var, name, label, cmap, color] 

# ----------------------------------------------------------------------- conv_org_metrics ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['conv_org',       f'obj_snapshot{c_prctile(prctile)}{t_type(switchM)}',         'obj [binary]',                       cmap,       color]    if metric in ['obj']          else [var, name, label, cmap, color]      
    var, name, label, cmap, color = ['conv_org',       f'rome{c_prctile(prctile)}{t_type(switchM)}',                 r'ROME [km$^2$]',                     cmap,       color]    if metric in ['rome']         else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['conv_org',       f'rome_n{c_prctile(prctile)}{t_type(switchM)}',               r'ROME [km$^2$]',                     cmap,       color]    if metric in ['rome_n']       else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['conv_org',       f'ni{c_prctile(prctile)}{t_type(switchM)}',                   'number index [Nb]',                  cmap,       color]    if metric in ['ni']           else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['conv_org',       f'areafraction{c_prctile(prctile)}{t_type(switchM)}',         'area frac. [N%]',                    cmap,       color]    if metric in ['areafraction'] else [var, name, label, cmap, color]   
    var, name, label, cmap, color = ['conv_org',       f'mean_area{c_prctile(prctile)}{t_type(switchM)}',             r'area [km$^2$]',                     cmap,       color]    if metric in ['mean_area']    else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['conv_org',       f'F_pr10',                                                     r'area [%]',                          cmap,       color]    if metric in ['F_pr10']       else [var, name, label, cmap, color]   
    var, name, label, cmap, color = ['conv_org',       f'pwad',                                                       r'pwad [%]',                          cmap,       color]    if metric in ['pwad']         else [var, name, label, cmap, color]   

# ----------------------------------------------------------------------- ls-state_metrics ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['ecs',       'ecs',                                                       'ECS [K]',                          'Reds',     'r']      if metric in ['ecs']            else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['tas',       f'tas{h_reg(switchM)}{m_type(switchM)}',                       r'tas [$\degree$C]',                'Reds',     'r']      if metric in ['tas']            else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['tas',       f'oni',                                                      r'oni [$\degree$C]',                'Reds',     'r']      if metric in ['oni']            else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['stability', f'stability{h_reg(switchM)}{m_type(switchM)}',                 r'stability [K]',                   'Reds',     'k']      if metric in ['stability']      else [var, name, label, cmap, color]

    var, name, label, cmap, color = ['hur',       f'hur{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',        'rel. humid. [%]',                    'Greens',   'g']      if metric in ['hur']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['hus',       f'hus{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',        'spec. humid. [%]',                   'Greens',   'g']      if metric in ['hus']          else [var, name, label, cmap, color]

    var, name, label, cmap, color = ['rlds',      f'rlds{h_reg(switchM)}{m_type(switchM)}',                      r'LW dwel. surf. [W m$^{-2}$]',       'Purples',    'purple'] if metric in ['rlds']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rlus',      f'rlus{h_reg(switchM)}{m_type(switchM)}',                      r'LW upwel. surf. [W m$^{-2}$]',      'Purples',    'purple'] if metric in ['rlus']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rlut',      f'rlut{h_reg(switchM)}{m_type(switchM)}',                      r'OLR [W m$^{-2}$]',                  'Purples',    'purple'] if metric in ['rlut']         else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['netlw',     f'netlw{h_reg(switchM)}{m_type(switchM)}',                     r'NetlW [W m$^{-2}$]',                'Purples',    'purple'] if metric in ['netlw']        else [var, name, label, cmap, color]    

    var, name, label, cmap, color = ['rsdt',      f'rsdt{h_reg(switchM)}{m_type(switchM)}',                      r'SW dwel TOA [W m$^{-2}$]',          'Purples',    'purple'] if metric in ['rsdt']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rsds',      f'rsds{h_reg(switchM)}{m_type(switchM)}',                      r'SW dwel surf. [W m$^{-2}$]',        'Purples',    'purple'] if metric in ['rsds']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rsus',      f'rsus{h_reg(switchM)}{m_type(switchM)}',                      r'SW upwel surf. [W m$^{-2}$]',       'Purples',    'purple'] if metric in ['rsus']         else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['rsut',      f'rsut{h_reg(switchM)}{m_type(switchM)}',                      r'SW upwel TOA [W m$^{-2}$]',         'Purples',    'purple'] if metric in ['rsut']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['netsw',     f'netsw{h_reg(switchM)}{m_type(switchM)}',                     r'NetSW [W m$^{-2}$]',                'Purples',    'purple'] if metric in ['netsw']        else [var, name, label, cmap, color]       

    var, name, label, cmap, color = ['h',        f'h{v_reg(switchM)}{v_reg(switchM)}{m_type(switchM)}',          r'h [J/kg]',                          'Blues',              'b'] if metric in ['h']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['h_anom2',  f'h_anom2{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',    r'h [(J/kg)$^{2}$]',                  'Blues',              'b'] if metric in ['h_anom2']    else [var, name, label, cmap, color]

    var, name, label, cmap, color = ['lcf',       f'lcf{h_reg(switchM)}{m_type(switchM)}',                       'low cloud frac. [%]',                  'Blues',    'b']    if metric in ['lcf']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['hcf',       f'hcf{h_reg(switchM)}{m_type(switchM)}',                       'high cloud frac. [%]',                 'Blues',    'b']    if metric in ['hcf']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['ws',        f'ws_lc{m_type(switchM)}',                                    r'weather state freq. [Nb day$^{-1}$]', 'Blues',    'b']    if metric in ['ws_lc']        else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['ws',        f'ws_hc{m_type(switchM)}',                                    r'weather state freq. [Nb day$^{-1}$]', 'Blues',    'b']    if metric in ['ws_hc']        else [var, name, label, cmap, color]

    var, name, label, cmap, color = ['wap',       f'wap{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',      r'wap [hPa day$^{-1}$]',              'RdBu_r',   'k']      if metric in ['wap']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['wap',       f'wap{v_reg(switchM)}_itcz_width{m_type(switchM)}',           r'lat [degrees]',                     'RdBu_r',   'k']      if metric in ['itcz_width']   else [var, name, label, cmap, color]

# ----------------------------------------------------------------------- conv_obj metrics ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['pr',        f'pr_o{m_type(switchM)}{c_prctile(prctile)}{t_type(switchM)}', r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_o']         else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['conv_org',       f'o_area{c_prctile(prctile)}{t_type(switchM)}',               r'area [km$^2$]',                     cmap,       color]    if metric in ['o_area']       else [var, name, label, cmap, color]  
    var, name, label, cmap, color = ['conv_org',       f'o_heatmap{c_prctile(prctile)}',                              r'freq [%]',                          cmap,       color]    if metric in ['o_heatmap']         else [var, name, label, cmap, color]   


# ------------------------------------------------------------------------- model_feat_calc ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['res',        'res',                                                      r'dlat x dlon [$\degree$]',            'Greys',    'k']     if metric in ['res']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['dlat',       'dlat',                                                     r'dlat [$\degree$]',                   'Greys',    'k']     if metric in ['dlat']         else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['dlon',       'dlon',                                                     r'dlon [$\degree$]',                   'Greys',    'k']     if metric in ['dlon']         else [var, name, label, cmap, color]

# ------------------------------------------------------------------- Exceptions / set plot features manually ----------------------------------------------------------------------------------------------------- #
    cmap, label = cmap_label_exceptions(metric, switchM, cmap, label)
    # cmap, color = 'Oranges', 'k' 
    label = 'area [%]' if m_type(switchM)=='_area_pos' else label
    return metric_class(var, name, cmap, label, color)



# -----------------------
#     Get metric
# -----------------------
def get_folder(folder_parent, met_type, metric_name, source):
    return f'{folder_parent}/metrics/{met_type}/{metric_name}/{source}'

def get_filename(dataset, experiment, metric_name, source, timescale = mV.timescales[0]):
    filename = f'{dataset}_{metric_name}_{timescale}_{experiment}_{mV.resolutions[0]}'
    if source in ['obs']:  
        filename = f'{dataset}_{metric_name}_{timescale}_{mV.obs_years[0]}_{experiment}_{mV.resolutions[0]}'
    # print(metric_name)
    if mV.resolutions[0] == 'regridded': #and metric_name in ['rome_95thprctile', 'o_area_95thprctile', 'pr_o_95thprctile']:
        filename = f'{filename}_{int(360/mV.x_res)}x{int(180/mV.y_res)}'
    # print(f'filename is: {filename}')
    return filename

def save_metric(switch, met_type, dataset, experiment, metric, metric_name):
    ''' Saves in variable/metric specific folders '''
    source = mF.find_source(dataset)
    filename = get_filename(dataset, experiment, metric_name, source)
    for save_type in [k for k, v in switch.items() if v]:
        folder = f'{home}/Desktop/{metric_name}'                                if save_type == 'save_folder_desktop'   else None
        folder = get_folder(mV.folder_scratch, met_type, metric_name, source)   if save_type == 'save_scratch'          else folder
        folder = get_folder(mV.folder_save, met_type, metric_name, source)      if save_type == 'save'                  else folder
        if not folder == None:
            mF.save_file(xr.Dataset({metric_name: metric}), folder, f'{filename}.nc')
            print(f'\t\t\t{metric_name} {save_type}')
            return f'{folder}/{filename}.nc'

def load_metric(metric_obj, dataset = 'TaiESM1', experiment = 'historical'):
    source = mF.find_source(dataset)
    if metric_obj.var in ['pr', 'conv_org'] and source in ['obs']:                                                               # GPCP observations used for precipitation based metrics
        dataset = 'GPCP'    
    filename = get_filename(dataset, experiment, metric_name = metric_obj.name, source = source)
    file_paths = []
    for folder_parent in [mV.folder_scratch, mV.folder_save]:                                                               # check for data in different folders
        folder = get_folder(folder_parent, met_type = metric_obj.var, metric_name = metric_obj.name, source = source)                                                  
        filename = get_filename(dataset, experiment, metric_name=metric_obj.name, source=source)            
        file_paths.append(f'{folder}/{filename}.nc')
        filename_daily = get_filename(dataset, experiment, metric_name=metric_obj.name, source=source, timescale='daily')   # check variation of filename
        file_paths.append(f'{folder}/{filename_daily}.nc')
    # print(file_paths)
    for path in file_paths:
        try:
            ds = xr.open_dataset(path)
            return ds[metric_obj.name]
        except FileNotFoundError:
            continue

def load_metric_file(metric_type, metric_name, dataset, experiment):
    source = mF.find_source(dataset)
    if metric_type in ['pr', 'conv_org'] and source in ['obs']:                                                               # GPCP observations used for precipitation based metrics
        dataset = 'GPCP'    
    filename = get_filename(dataset, experiment, metric_name, source = source)
    file_paths = []
    for folder_parent in [mV.folder_scratch, mV.folder_save]:                                                               # check for data in different folders
        folder = get_folder(folder_parent, met_type = metric_type, metric_name = metric_name, source = source)                                                  
        filename = get_filename(dataset, experiment, metric_name, source)            
        file_paths.append(f'{folder}/{filename}.nc')
        filename_daily = get_filename(dataset, experiment, metric_name, source, timescale='daily')                          # check variation of filename
        file_paths.append(f'{folder}/{filename_daily}.nc')
    # if dataset == 'GPCP':
    # print(file_paths)
    dataset_loaded = False
    for path in file_paths:
        try:
            ds = xr.open_dataset(path)
            dataset_loaded = True 
            return ds[metric_name]
        except FileNotFoundError:
            continue
    if not dataset_loaded:
        raise FileNotFoundError(f"Failed to load metric file for {metric_name}, {dataset}, {experiment} from any provided paths.")


# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = mV.datasets[0]
    experiment = mV.experiments[0]
    source = mF.find_source(dataset)

    metric_class_option = False
    if metric_class_option:
        metric_obj = get_metric_class(metric = 'itcz_width', switchM = {'500hpa': True, 'sMean': True})
        folder= get_folder(folder_parent = mV.folder_scratch, met_type = metric_obj.var, metric_name = metric_obj.name, source = source)
        filename = get_filename(dataset, experiment, metric_name = metric_obj.name, source = source, timescale = mV.timescales[0])
        print(folder)
        print(filename)
        da = load_metric(metric_obj, dataset, experiment)
        print(da)
    
    metric_name_option = False
    if metric_name_option:
        # ds = xr.open_dataset(f'/scratch/w40/cb4968/metrics/wap/wap_500hpa_itcz_width/cmip6/ACCESS-CM2_wap_500hpa_itcz_width_monthly_historical_regridded_144x72.nc')
        metric_type = 'wap'
        metric_name = 'wap_500hpa_itcz_width'
        da = load_metric_file(metric_type, metric_name, dataset, experiment)
        print(da.data)


    # plot_object = da.plot()
    # fig = plot_object[0].figure
    # fig.savefig(f'{os.getcwd()}/test.png')
    # plt.show()




# '/scratch/w40/cb4968/metrics/wap/wap_500hpa_itcz_width/cmip6/'
# 'ACCESS-CM2_wap_500hpa_itcz_width_monthly_historical_regridded_144x72.nc'


