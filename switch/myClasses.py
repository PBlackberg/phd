import numpy as np



# ------------------------
#       Classes
# ------------------------
# --------------------------------------------------------------- Gives metric specs ----------------------------------------------------------------------------------------------------- #
class metric_class():
    ''' Gives metric: name (of saved dataset), option (data array in dataset), label, cmap, color (used for plots of calculated metrics) '''
    def __init__(self, var, met_name, cmap, label, color='k'):
        self.var           = var
        self.name          = met_name
        self.label         = label
        self.cmap          = cmap
        self.color         = color


# ----------------------------------------------------------------- Gives Variable specs ----------------------------------------------------------------------------------------------------- #
class variable_class():
    ''' Gives variable details (name, option, label, cmap) (Used for animation of fields) '''
    def __init__(self, ref, variable_type, name, cmap, label):
        self.ref           = ref
        self.variable_type = variable_type
        self.name          = name
        self.label         = label
        self.cmap          = cmap


# ------------------------------------------------------------------ Gives dimension specs ----------------------------------------------------------------------------------------------------- #
class dims_class():
    R = 6371        # radius of earth
    g = 9.81        # gravitaional constant
    c_p = 1.005     # specific heat capacity
    L_v = 2.256e6   # latent heat of vaporization
    def __init__(self, da):
        self.lat, self.lon       = da['lat'].data, da['lon'].data
        self.lonm, self.latm     = np.meshgrid(self.lon, self.lat)
        self.dlat, self.dlon     = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
        self.aream               = np.cos(np.deg2rad(self.latm))*np.float64(self.dlon*self.dlat*self.R**2*(np.pi/180)**2) # area of domain
        self.latm3d, self.lonm3d = np.expand_dims(self.latm,axis=2), np.expand_dims(self.lonm,axis=2)                     # used for broadcasting
        self.aream3d             = np.expand_dims(self.aream,axis=2)



# ------------------------
#  Variations of metric
# ------------------------
# ------------------------------------------------------------------------- metric type ----------------------------------------------------------------------------------------------------- #
def m_type(switch = {'snapshot':True}):
    ''' Metric type '''
    metric_type = ''
    for met_type in [k for k, v in switch.items() if v]:
        metric_type = '_snapshot' if met_type in ['snapshot'] else metric_type
        metric_type = '_sMean'    if met_type in ['sMean']    else metric_type
        metric_type = '_tMean'    if met_type in ['tMean']    else metric_type
        metric_type = '_area'     if met_type in ['area']     else metric_type
        metric_type = ''          if met_type in ['other']    else metric_type
    return metric_type


# ----------------------------------------------------------------------- Convective threshold ----------------------------------------------------------------------------------------------------- #
def c_prctile(prctile = '95'):
    ''' Convective percentile '''
    if prctile:
        prctile = f'_{prctile}thprctile'
    return prctile

def t_type(switch = {'fixed area':False}):
    ''' Convective threshold type '''
    threshold = ''
    for met_type in [k for k, v in switch.items() if v]:
        threshold = '_fixed_area' if met_type in ['fixed area'] else threshold
    return threshold


# -------------------------------------------------------------------------- vertical region ----------------------------------------------------------------------------------------------------- #
def v_reg(switch = {'a':False}):
    ''' Vertical region '''
    region = ''
    for met_type in [k for k, v in switch.items() if v]:
        region = '_250hpa' if met_type in ['250hpa'] else region
        region = '_700hpa' if met_type in ['700hpa'] else region
        region = '_500hpa' if met_type in ['500hpa'] else region
    return region


# -------------------------------------------------------------------------- horizontal region ----------------------------------------------------------------------------------------------------- #
def h_reg(switch = {'a':False}):
    ''' Horizontal region '''
    region = ''
    for met_type in [k for k, v in switch.items() if v]:
        region = '_d'           if met_type in ['descent']          else region
        region = '_a'           if met_type in ['ascent']           else region
        region = '_fd'          if met_type in ['descent_fixed']    else region
        region = '_fa'          if met_type in ['ascent_fixed']     else region
        region = f'_o{region}'  if met_type in ['ocean']            else region
    return region


# ------------------------------------------------------------------------------ exceptions ----------------------------------------------------------------------------------------------------- #
def cmap_label_exceptions(metric, switch, cmap, label):
    ''' Change cmap
    RbBu        - Change with warming 
    Blue / Red  - wap_reg '''
    for met_type in [k for k, v in switch.items() if v]:
        cmap = 'RdBu_r' if met_type in ['change with warming'] else cmap
        cmap = 'Reds'  if met_type in ['descent'] and metric in ['wap'] else cmap   # wap normally has positive and negative values, but not in descent / ascent
        cmap = 'Blues' if met_type in ['ascent']  and metric in ['wap'] else cmap

    for met_type in [k for k, v in switch.items() if v]:                            # add per K on change with warming
        label = '{}{}{}'.format(label[:-1], r'K$^{-1}$', label[-1:]) if met_type in ['per kelvin', 'per ecs'] and not label == 'ECS [K]' else label 
    return cmap, label



# -----------------------
#      All metrics
# -----------------------
def get_metric_class(metric, switchM = {'metric_variation':False}, prctile = '95'):
    ''' Used for loading and plotting metrics '''
    var, name, label, cmap, color = [None, None, 'test', 'Greys', 'k']

# ------------------------------------------------------------------- Precipitation ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['pr',        f'pr{m_type(switchM)}',                                        r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr']           else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['pr',        f'pr_90{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_90']        else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['pr',        f'pr_95{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_95']        else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['pr',        f'pr_97{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_97']        else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_99{m_type(switchM)}',                                     r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_99']        else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_rx1day{m_type(switchM)}',                                 r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_rx1day']    else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_rx5day{m_type(switchM)}',                                 r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_rx5day']    else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['pr',        f'pr_o{m_type(switchM)}{c_prctile(prctile)}{t_type(switchM)}',  r'pr [mm day$^{-1}$]',                'Blues',    'b']      if metric in ['pr_o']         else [var, name, label, cmap, color]


# -------------------------------------------------------------------- Organization ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['org',       f'obj_snapshot{c_prctile(prctile)}{t_type(switchM)}',         'obj [binary]',                       cmap,       color]    if metric in ['obj']          else [var, name, label, cmap, color]      
    var, name, label, cmap, color = ['org',       f'rome{c_prctile(prctile)}{t_type(switchM)}',                 r'ROME [km$^2$]',                     cmap,       color]    if metric in ['rome']         else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['org',       f'rome_n{c_prctile(prctile)}{t_type(switchM)}',               r'ROME [km$^2$]',                     cmap,       color]    if metric in ['rome_n']       else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['org',       f'ni{c_prctile(prctile)}{t_type(switchM)}',                   'number index [Nb]',                  cmap,       color]    if metric in ['ni']           else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['org',       f'areafraction{c_prctile(prctile)}{t_type(switchM)}',         'area frac. [N%]',                    cmap,       color]    if metric in ['areafraction'] else [var, name, label, cmap, color]   
    var, name, label, cmap, color = ['org',       f'o_area{c_prctile(prctile)}{t_type(switchM)}',               r'area [km$^2$]',                     cmap,       color]    if metric in ['o_area']       else [var, name, label, cmap, color]  
    var, name, label, cmap, color = ['org',       f'mean_area{c_prctile(prctile)}{t_type(switchM)}',             r'area [km$^2$]',                     cmap,       color]    if metric in ['mean_area']    else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['org',       f'F_pr10',                                                     r'area [%]',                          cmap,       color]    if metric in ['F_pr10']       else [var, name, label, cmap, color]   
    var, name, label, cmap, color = ['org',       f'pwad',                                                       r'pwad [%]',                          cmap,       color]    if metric in ['pwad']         else [var, name, label, cmap, color]   
    var, name, label, cmap, color = ['org',       f'o_heatmap{c_prctile(prctile)}',                              r'freq [%]',                          cmap,       color]    if metric in ['o_heatmap']         else [var, name, label, cmap, color]   

# --------------------------------------------------------------------- Temperature ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['ecs',       'ecs',                                                       'ECS [K]',                          'Reds',     'r']      if metric in ['ecs']            else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['tas',       f'tas{h_reg(switchM)}{m_type(switchM)}',                       r'tas [$\degree$C]',                'Reds',     'r']      if metric in ['tas']            else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['stability', f'stability{h_reg(switchM)}{m_type(switchM)}',                 r'stability [K]',                   'Reds',     'k']      if metric in ['stability']      else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['tas',       f'oni',                                                      r'oni [$\degree$C]',                'Reds',     'r']      if metric in ['oni']            else [var, name, label, cmap, color]

# --------------------------------------------------------------------- Circulation ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['wap',       f'wap{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',        r'wap [hPa day$^{-1}$]',              'RdBu_r',   'k']      if metric in ['wap']          else [var, name, label, cmap, color]
 

# ----------------------------------------------------------------------- Humidity ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['hur',       f'hur{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',        'rel. humid. [%]',                    'Greens',   'g']      if metric in ['hur']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['hus',       f'hus{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',        'spec. humid. [%]',                   'Greens',   'g']      if metric in ['hus']          else [var, name, label, cmap, color]


# ----------------------------------------------------------------------- Radiation ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['rlds',      f'rlds{h_reg(switchM)}{m_type(switchM)}',                      r'LW dwel. surf. [W m$^{-2}$]',       'Purples',    'purple'] if metric in ['rlds']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rlus',      f'rlus{h_reg(switchM)}{m_type(switchM)}',                      r'LW upwel. surf. [W m$^{-2}$]',      'Purples',    'purple'] if metric in ['rlus']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rlut',      f'rlut{h_reg(switchM)}{m_type(switchM)}',                      r'OLR [W m$^{-2}$]',                  'Purples',    'purple'] if metric in ['rlut']         else [var, name, label, cmap, color]    
    var, name, label, cmap, color = ['netlw',     f'netlw{h_reg(switchM)}{m_type(switchM)}',                     r'NetlW [W m$^{-2}$]',                'Purples',    'purple'] if metric in ['netlw']        else [var, name, label, cmap, color]    

    var, name, label, cmap, color = ['rsdt',      f'rsdt{h_reg(switchM)}{m_type(switchM)}',                      r'SW dwel TOA [W m$^{-2}$]',          'Purples',    'purple'] if metric in ['rsdt']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rsds',      f'rsds{h_reg(switchM)}{m_type(switchM)}',                      r'SW dwel surf. [W m$^{-2}$]',        'Purples',    'purple'] if metric in ['rsds']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['rsus',      f'rsus{h_reg(switchM)}{m_type(switchM)}',                      r'SW upwel surf. [W m$^{-2}$]',       'Purples',    'purple'] if metric in ['rsus']         else [var, name, label, cmap, color] 
    var, name, label, cmap, color = ['rsut',      f'rsut{h_reg(switchM)}{m_type(switchM)}',                      r'SW upwel TOA [W m$^{-2}$]',         'Purples',    'purple'] if metric in ['rsut']         else [var, name, label, cmap, color]       
    var, name, label, cmap, color = ['netsw',     f'netsw{h_reg(switchM)}{m_type(switchM)}',                     r'NetSW [W m$^{-2}$]',                'Purples',    'purple'] if metric in ['netsw']        else [var, name, label, cmap, color]       


# -------------------------------------------------------------------- Moist Static Energy ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['h',        f'h{v_reg(switchM)}{v_reg(switchM)}{m_type(switchM)}',          r'h [J/kg]',                            'Blues',              'b'] if metric in ['h']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['h_anom2',  f'h_anom2{v_reg(switchM)}{h_reg(switchM)}{m_type(switchM)}',    r'h [(J/kg)$^{2}$]',                    'Blues',              'b'] if metric in ['h_anom2']    else [var, name, label, cmap, color]


# ------------------------------------------------------------------------- Clpids ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['lcf',       f'lcf{h_reg(switchM)}{m_type(switchM)}',                       'low cloud frac. [%]',                  'Blues',    'b']    if metric in ['lcf']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['hcf',       f'hcf{h_reg(switchM)}{m_type(switchM)}',                       'high cloud frac. [%]',                 'Blues',    'b']    if metric in ['hcf']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['ws',        f'ws_lc{m_type(switchM)}',                                    r'weather state freq. [Nb day$^{-1}$]', 'Blues',    'b']    if metric in ['ws_lc']        else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['ws',        f'ws_hc{m_type(switchM)}',                                    r'weather state freq. [Nb day$^{-1}$]', 'Blues',    'b']    if metric in ['ws_hc']        else [var, name, label, cmap, color]


# --------------------------------------------------------------------------- Other ----------------------------------------------------------------------------------------------------- #
    var, name, label, cmap, color = ['res',        'res',                                                      r'dlat x dlon [$\degree$]',            'Greys',    'k']     if metric in ['res']          else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['dlat',       'dlat',                                                     r'dlat [$\degree$]',                   'Greys',    'k']     if metric in ['dlat']         else [var, name, label, cmap, color]
    var, name, label, cmap, color = ['dlon',       'dlon',                                                     r'dlon [$\degree$]',                   'Greys',    'k']     if metric in ['dlon']         else [var, name, label, cmap, color]


# ------------------------------------------------------------------- Exceptions / set manually ----------------------------------------------------------------------------------------------------- #
    cmap, label = cmap_label_exceptions(metric, switchM, cmap, label)
    # cmap, color = 'Oranges', 'k' 
    label = 'area [%]' if m_type(switchM)=='_area' else label
    return metric_class(var, name, cmap, label, color)



# -----------------------
#     All variables
# -----------------------
# ----------------------------------------------------------------------------------- variable specs (for field animations) ----------------------------------------------------------------------------------------------------- #
def get_variable_class(switch):
    ''' list of variable: name (of saved dataset), option (data array in dataset), label, cmap, color. Used for animation of fields '''
    ref, var, name, label, cmap = [None, None, None, None, None]
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        ref, var, name, label, cmap = [key, 'pr', 'pr',   r'pr [mm day${^-1}$]',  'Greys']    if key in ['obj']  else [ref, var, name, label, cmap] 
        ref, var, name, label, cmap = [key, key,  key,    r'pr [mm day${^-1}$]',  'Blues']    if key in ['pr']   else [ref, var, name, label, cmap] 
        ref, var, name, label, cmap = [key, 'pr', 'pr',   r'pr [mm day${^-1}$]',  'Reds']     if key in ['pr99'] else [ref, var, name, label, cmap] 
        ref, var, name, label, cmap = [key, key,  key,    'rel. humiid. [%]',     'Greens']   if key in ['hur']  else [ref, var, name, label, cmap] 
        ref, var, name, label, cmap = [key, 'rad', key,   r'OLR [W m${^-2}$]',    'Purples'] if key in ['rlut']  else [ref, var, name, label, cmap]
    return variable_class(ref, var, name, cmap, label)























