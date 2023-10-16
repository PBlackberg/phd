import numpy as np

# ---------------------------------------------------------------------------- Variation of metric ----------------------------------------------------------------------------------------------------- #
def m_type(switch):
    metric_type = ''
    for met_type in [k for k, v in switch.items() if v]:
        metric_type = '_snapshot' if met_type in ['snapshot'] else metric_type
        metric_type = '_sMean'    if met_type in ['sMean']    else metric_type
        metric_type = '_tMean'    if met_type in ['tMean']    else metric_type
        metric_type = '_area'     if met_type in ['area']     else metric_type
        metric_type = ''          if met_type in ['other']    else metric_type
    return metric_type

def c_prctile(prctile):
    ''' picks precipitation percentile '''
    if prctile:
        prctile = f'_{prctile}thprctile'
    return prctile

def t_type(switch):
    ''' picks type of threshold. One area on on precipitation rate '''
    threshold = ''
    for met_type in [k for k, v in switch.items() if v]:
        threshold = '_fixed_area' if met_type in ['fixed area'] else threshold
    return threshold

def v_reg(switch):
    ''' picks region of ascent / descent '''
    region = ''
    for met_type in [k for k, v in switch.items() if v]:
        region = '_250hpa' if met_type in ['250hpa'] else region
        region = '_700hpa' if met_type in ['700hpa'] else region
        region = '_500hpa' if met_type in ['500hpa'] else region
    return region

def h_reg(switch):
    ''' picks region of ascent / descent '''
    region = ''
    region = '_d' if switch['descent'] else region
    region = '_a' if switch['ascent']  else region
    region = f'{region}_o' if switch['ocean']  else region
    return region

def exceptions(metric, switch, cmap, label):
    for met_type in [k for k, v in switch.items() if v]:
        cmap = 'RdBu_r' if met_type in ['change with warming'] else cmap
        # cmap = 'Reds'  if met_type in ['descent'] and metric in ['wap'] else cmap
        # cmap = 'Blues' if met_type in ['ascent']  and metric in ['wap'] else cmap

    for met_type in [k for k, v in switch.items() if v]:
        label = '{}{}{}'.format(label[:-1], r'K$^{-1}$', label[-1:]) if met_type in ['per kelvin'] and not label == 'ECS [K]' else label
    return cmap, label



# ----------------------------------------------------------------------------------- metric specs (for plots) ----------------------------------------------------------------------------------------------------- #
class metric_class():
    ''' Gives metric: name (of saved dataset), option (data array in dataset), label, cmap, color (used for plots of calculated metrics) '''
    def __init__(self, var_type, met_name, cmap, label, color='k'):
        self.var_type      = var_type
        self.name          = met_name
        self.label         = label
        self.cmap          = cmap
        self.color         = color

def get_metric_class(metric, switch = {'a':False}, prctile = '95'):
    ''' list of metric: name (of saved dataset), option (data array in dataset), label, cmap, color. Used for plots of metrics '''
    var_type, met_name, label, cmap, color = [None, None, None, 'Greys', 'k']
    # -------------
    # precipitation
    # -------------
    var_type, met_name, label, cmap, color = ['pr',        f'pr{m_type(switch)}',                                        r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr']           else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['pr',        f'pr_90{m_type(switch)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_90']        else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['pr',        f'pr_95{m_type(switch)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_95']        else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['pr',        f'pr_97{m_type(switch)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_97']        else [var_type, met_name, label, cmap, color] 
    var_type, met_name, label, cmap, color = ['pr',        f'pr_99{m_type(switch)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_99']        else [var_type, met_name, label, cmap, color] 
    var_type, met_name, label, cmap, color = ['pr',        f'pr_rx1day{m_type(switch)}',                                 r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_rx1day']    else [var_type, met_name, label, cmap, color] 
    var_type, met_name, label, cmap, color = ['pr',        f'pr_rx5day{m_type(switch)}',                                 r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_rx5day']    else [var_type, met_name, label, cmap, color] 
    var_type, met_name, label, cmap, color = ['pr',        f'pr_o{m_type(switch)}{c_prctile(prctile)}{t_type(switch)}',  r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_o']         else [var_type, met_name, label, cmap, color]


    # ------------
    # organization
    # ------------
    var_type, met_name, label, cmap, color = ['org',       f'obj_snapshot{c_prctile(prctile)}{t_type(switch)}',         'obj [binary]',                       cmap,       color]    if metric in ['obj']          else [var_type, met_name, label, cmap, color]      
    var_type, met_name, label, cmap, color = ['org',       f'rome{c_prctile(prctile)}{t_type(switch)}',                 r'ROME [km$^2$]',                     cmap,       color]    if metric in ['rome']         else [var_type, met_name, label, cmap, color]    
    var_type, met_name, label, cmap, color = ['org',       f'rome_n{c_prctile(prctile)}{t_type(switch)}',               r'ROME [km$^2$]',                     cmap,       color]    if metric in ['rome_n']       else [var_type, met_name, label, cmap, color]    
    var_type, met_name, label, cmap, color = ['org',       f'ni{c_prctile(prctile)}{t_type(switch)}',                   'number index [Nb]',                  cmap,       color]    if metric in ['ni']           else [var_type, met_name, label, cmap, color]    
    var_type, met_name, label, cmap, color = ['org',       f'areafraction{c_prctile(prctile)}{t_type(switch)}',         'area frac. [N%]',                    cmap,       color]    if metric in ['areafraction'] else [var_type, met_name, label, cmap, color]   
    var_type, met_name, label, cmap, color = ['org',       f'o_area{c_prctile(prctile)}{t_type(switch)}',               r'area [km$^2$]',                     cmap,       color]    if metric in ['o_area']       else [var_type, met_name, label, cmap, color]  
    var_type, met_name, label, cmap, color = ['org',       f'mean_area{c_prctile(prctile)}{t_type(switch)}',            r'area [km$^2$]',                     cmap,       color]    if metric in ['mean_area']    else [var_type, met_name, label, cmap, color]    
    var_type, met_name, label, cmap, color = ['org',       f'F_pr10{c_prctile(prctile)}{t_type(switch)}',               r'area [km$^2$]',                     cmap,       color]    if metric in ['F_pr10']       else [var_type, met_name, label, cmap, color]   


    # ------------
    # Temperature
    # ------------
    var_type, met_name, label, cmap, color = ['ecs',       'ecs',                                                       'ECS [K]',                            'Reds',     'r']      if metric in ['ecs']          else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['tas',       f'tas{h_reg(switch)}{m_type(switch)}',                       r'temp. [$\degree$C]',                'Reds',     'r']      if metric in ['tas']          else [var_type, met_name, label, cmap, color]
    

    # ------------
    #  Drying
    # ------------
    var_type, met_name, label, cmap, color = ['wap',       f'wap{v_reg(switch)}{h_reg(switch)}{m_type(switch)}',        r'wap [hPa day$^{-1}$]',              'RdBu_r',   'k']      if metric in ['wap']          else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['hur',       f'hur{v_reg(switch)}{h_reg(switch)}{m_type(switch)}',        'rel. humid. [%]',                    'Greens',   'g']      if metric in ['hur']          else [var_type, met_name, label, cmap, color]


    # ------------
    #  Radiation
    # ------------
    var_type, met_name, label, cmap, color = ['rlds',      f'rlds{h_reg(switch)}{m_type(switch)}',                      r'LW dwel. surf. [W m$^{-2}$]',       'Purples',  'purple'] if metric in ['rlds']         else [var_type, met_name, label, cmap, color]       
    var_type, met_name, label, cmap, color = ['rlus',      f'rlus{h_reg(switch)}{m_type(switch)}',                      r'LW upwel. surf. [W m$^{-2}$]',      'Purples',  'purple'] if metric in ['rlus']         else [var_type, met_name, label, cmap, color]       
    var_type, met_name, label, cmap, color = ['rlut',      f'rlut{h_reg(switch)}{m_type(switch)}',                      r'OLR [W m$^{-2}$]',                  'Purples',  'purple'] if metric in ['rlut']         else [var_type, met_name, label, cmap, color]    
    var_type, met_name, label, cmap, color = ['netlw',     f'netlw{h_reg(switch)}{m_type(switch)}',                     r'NetlW [W m$^{-2}$]',                'Purples',  'purple'] if metric in ['netlw']        else [var_type, met_name, label, cmap, color]    

    var_type, met_name, label, cmap, color = ['rsdt',      f'rsdt{h_reg(switch)}{m_type(switch)}',                      r'SW dwel TOA [W m$^{-2}$]',          'Purples',  'purple'] if metric in ['rsdt']         else [var_type, met_name, label, cmap, color]       
    var_type, met_name, label, cmap, color = ['rsds',      f'rsds{h_reg(switch)}{m_type(switch)}',                      r'SW dwel surf. [W m$^{-2}$]',        'Purples',  'purple'] if metric in ['rsds']         else [var_type, met_name, label, cmap, color]       
    var_type, met_name, label, cmap, color = ['rsus',      f'rsus{h_reg(switch)}{m_type(switch)}',                      r'SW upwel surf. [W m$^{-2}$]',       'Purples',  'purple'] if metric in ['rsus']         else [var_type, met_name, label, cmap, color] 
    var_type, met_name, label, cmap, color = ['rsut',      f'rsut{h_reg(switch)}{m_type(switch)}',                      r'SW upwel TOA [W m$^{-2}$]',         'Purples',  'purple'] if metric in ['rsut']         else [var_type, met_name, label, cmap, color]       
    var_type, met_name, label, cmap, color = ['netsw',     f'netsw{h_reg(switch)}{m_type(switch)}',                     r'NetSW [W m$^{-2}$]',                'Purples',  'purple'] if metric in ['netsw']        else [var_type, met_name, label, cmap, color]       


    # ---------
    #  clouds
    # ---------
    var_type, met_name, label, cmap, color = ['stability', f'stability{h_reg(switch)}{m_type(switch)}',                 r'temp. [hPa day$^{-1}$]',              'Reds',     'k']    if metric in ['stability']    else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['lcf',       f'lcf{v_reg(switch)}{h_reg(switch)}{m_type(switch)}',        'low cloud frac. [%]',                  'Blues',    'b']    if metric in ['lcf']          else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['hcf',       f'hcf{v_reg(switch)}{h_reg(switch)}{m_type(switch)}',        'high cloud frac. [%]',                 'Blues',    'b']    if metric in ['hcf']          else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['ws',        f'ws_lc{m_type(switch)}',                                    r'weather state freq. [Nb day$^{-1}$]', 'Blues',    'b']    if metric in ['ws_lc']        else [var_type, met_name, label, cmap, color]
    var_type, met_name, label, cmap, color = ['ws',        f'ws_hc{m_type(switch)}',                                    r'weather state freq. [Nb day$^{-1}$]', 'Blues',    'b']    if metric in ['ws_hc']        else [var_type, met_name, label, cmap, color]


    # -------------------
    # Moist static energy
    # -------------------
    var_type, met_name, label, cmap, color = ['hus',       f'hus{v_reg(switch)}{h_reg(switch)}{m_type(switch)}',        'spec. humid. [%]',                   'Greens',   'g']      if metric in ['hus']          else [var_type, met_name, label, cmap, color]


    # ---------------------------
    #  Exceptions / Manually set
    # ---------------------------
    cmap, label = exceptions(metric, switch, cmap, label)
    return metric_class(var_type, met_name, cmap, label, color)



# ----------------------------------------------------------------------------------- variable specs (for field animations) ----------------------------------------------------------------------------------------------------- #
class variable_class():
    ''' Gives variable details (name, option, label, cmap) (Used for animation of fields) '''
    def __init__(self, ref, variable_type, name, cmap, label):
        self.ref           = ref
        self.variable_type = variable_type
        self.name          = name
        self.label         = label
        self.cmap          = cmap

def get_variable_class(switch):
    ''' list of variable: name (of saved dataset), option (data array in dataset), label, cmap, color. Used for animation of fields '''
    ref, variable_type, name, label, cmap = [None, None, None, None, None]
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        ref, variable_type, name, label, cmap = [key, 'pr', 'pr',   r'pr [mm day${^-1}$]',  'Greys']    if key in ['obj']  else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, key,  key,    r'pr [mm day${^-1}$]',  'Blues']    if key in ['pr']   else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, 'pr', 'pr',   r'pr [mm day${^-1}$]',  'Reds']     if key in ['pr99'] else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, key,  key,    'rel. humiid. [%]',     'Greens']   if key in ['hur']  else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, 'rad', key,   r'OLR [W m${^-2}$]',    'Purples'] if key in ['rlut']  else [ref, variable_type, name, label, cmap]
    return variable_class(ref, variable_type, name, cmap, label)



# ----------------------------------------------------------------------------------- Dimensions of dataset ----------------------------------------------------------------------------------------------------- #
class dims_class():
    R = 6371
    def __init__(self, da):
        self.lat, self.lon  = da['lat'].data, da['lon'].data
        self.lonm, self.latm = np.meshgrid(self.lon, self.lat)
        self.dlat, self.dlon = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
        self.aream = np.cos(np.deg2rad(self.latm))*np.float64(self.dlon*self.dlat*self.R**2*(np.pi/180)**2) # used for area of object
        self.latm3d, self.lonm3d = np.expand_dims(self.latm,axis=2), np.expand_dims(self.lonm,axis=2) # used for broadcasting
        self.aream3d = np.expand_dims(self.aream,axis=2)



























