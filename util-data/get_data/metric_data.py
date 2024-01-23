# ------------------------
#    Metric class
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
#   Get metric object
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


def load_metric(metric_class, dataset = mV.datasets[0], experiment = mV.experiments[0], timescale = mV.timescales[0], resolution = mV.resolutions[0], folder_load = mV.folder_save[0]):
    source = find_source(dataset)
    experiment = ''             if source in ['obs'] else experiment
    if source in ['obs'] and dataset not in ['GPCP', 'GPCP_1998-2009', 'GPCP_2010-2022']:
        # dataset = 'GPCP_1998-2009'  if source == 'obs' and metric_class.var in ['pr', 'org'] else dataset  # for comparing with other obs datasets
        dataset = 'GPCP_2010-2022'  if source == 'obs' and metric_class.var in ['pr', 'org'] else dataset   # for comparing with other obs datasets
        # dataset = 'GPCP'            if source == 'obs' and metric_class.var in ['pr', 'org'] else dataset  # for comparing with other obs datasets
    timescale = 'daily'         if metric_class.var in ['pr', 'org', 'hus', 'ws'] else timescale            # some metrics are only on daily
    timescale = 'monthly'         if dataset == 'NOAA'    else timescale                                      # some datasets are only on daily
    timescale = 'daily'         if dataset == 'ISCCP'   else timescale                                      # some datasets are only on daily

    path = f'{folder_load}/metrics/{metric_class.var}/{metric_class.name}/{source}/{dataset}_{metric_class.name}_{timescale}_{experiment}_{resolution}.nc'
    ds = xr.open_dataset(path)     
    da = ds[f'{metric_class.name}']
    if dataset == 'CERES':
        da['time'] = da['time'] - pd.Timedelta(days=14) # this observational dataset have monthly data with day specified as the middle of the month instead of the first
    return da












