import numpy as np

def cloud_type(switch):
    ''' picks region of ascent / descent '''
    cloud_type = ''
    cloud_type = '_low_clouds' if switch['low_clouds'] else cloud_type
    cloud_type = '_high_clouds' if switch['high_clouds'] else cloud_type
    return cloud_type

def vert_reg(switch):
    ''' picks region of ascent / descent '''
    region = ''
    region = '_250hpa' if switch['250hpa'] else region
    region = '_700hpa' if switch['700hpa']  else region
    return region

def reg(switch):
    ''' picks region of ascent / descent '''
    region = ''
    region = '_d' if switch['descent'] else region
    region = '_a' if switch['ascent']  else region
    return region

def thres(switch):
    ''' picks precipitation threshold for organization metrics '''
    threshold = ''
    threshold = '_fixed_area' if switch['fixed area'] else threshold
    return threshold


# ----------------------------------------------------------------------------------- metric specs (for plots) ----------------------------------------------------------------------------------------------------- #
class metric_class():
    ''' Gives metric: name (of saved dataset), option (data array in dataset), label, cmap, color (used for plots of calculated metrics) '''
    def __init__(self, variable_type, name, option, cmap, label, color='k'):
        self.variable_type = variable_type
        self.name   = name
        self.option = option
        self.label  = label
        self.cmap   = cmap
        self.color  = color

def get_metric_object(switch):
    ''' list of metric: name (of saved dataset), option (data array in dataset), label, cmap, color. Used for plots of metrics '''
    variable_type, name, option, label, cmap, color = [None, None, None, None, 'Greys', 'k']
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        # -------------
        # precipitation
        # -------------
        variable_type, name, option, label, cmap, color = ['pr', 'pr',                   key, r'pr [$mm day{^-1}]',  'Blues', 'b'] if key in ['pr']                                   else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'rxday_pr',             key, r'pr [mm day$^{-1}$]', 'Blues', 'b'] if key in ['rx1day_pr','rx5day_pr']                else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'rxday_pr_sMean',       key, r'pr [mm day$^{-1}$]', 'Blues', 'b'] if key in ['rx1day_pr_sMean','rx5day_pr_sMean']    else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'percentiles_pr',       key, r'pr [mm day$^{-1}$]', 'Blues', 'b'] if key in ['pr95','pr97','pr99']                   else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'percentiles_pr_sMean', key, r'pr [mm day$^{-1}$]', 'Blues', 'b'] if key in ['pr95_sMean','pr97_sMean','pr99_sMean'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['pr', 'F_pr10',               key, 'F_pr10 [%]',           cmap,   color] if key in ['F_pr10']                             else [variable_type, name, option, label, cmap, color]


        # ------------
        # organization
        # ------------
        variable_type, name, option, label, cmap, color = ['org', f'{key}{thres(switch)}', key, 'obj [binary]',      cmap, color] if key in ['obj']            else [variable_type, name, option, label, cmap, color]          
        variable_type, name, option, label, cmap, color = ['org', f'{key}{thres(switch)}', key, r'ROME [km$^2$]',    cmap, color] if key in ['rome', 'rome_n'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['org', f'{key}{thres(switch)}', key, 'number index [Nb]', cmap, color] if key in ['ni']             else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['org', f'ni{thres(switch)}',    key, 'areafraction [%]',  cmap, color] if key in ['areafraction']   else [variable_type, name, option, label, cmap, color]


        # -----------------------------
        # large-scale environment state
        # -----------------------------
        variable_type, name, option, label, cmap, color = [None,  key,                                     key,                                     'ECS [K]',               'Reds',     'r']      if key in ['ecs']       else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,   f'{key}{reg(switch)}',                   f'{key}{reg(switch)}',                   r'temp. [$\degree$C]',   'coolwarm', 'r']      if key in ['tas']       else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,   f'{key}{vert_reg(switch)}{reg(switch)}', f'{key}{vert_reg(switch)}{reg(switch)}', 'rel. humid. [%]',       'Greens',   'g']      if key in ['hur']       else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,   f'{key}{vert_reg(switch)}{reg(switch)}', f'{key}{vert_reg(switch)}{reg(switch)}', r'wap [hPa day$^{-1}$]', 'RdBu_r',   color]    if key in ['wap']       else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['wap', f'{key}_a',                              f'{key}_a',                              'ascent area [%]',       'Greys',    'k']      if key in ['wap_area']  else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = [key,   f'{key}{reg(switch)}',                   f'{key}{reg(switch)}',                   'stability [K]',         'Reds',     color]    if key in ['stability'] else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['rad', f'{key}',                                f'{key}',                                r'OLR [W m${^-2}$]',     'Purples',  'purple'] if key in ['rlut']      else [variable_type, name, option, label, cmap, color]
                    

        # ---------
        #  clouds
        # ---------
        variable_type, name, option, label, cmap, color = ['cl', f'{key}{reg(switch)}',        f'{key}{reg(switch)}',        'low cloud fraction [%]',             'Blues', 'b'] if key == 'lcf' else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['cl', f'{key}{reg(switch)}',        f'{key}{reg(switch)}',        'high cloud fraction [%]',            'Blues', 'b'] if key == 'hcf' else [variable_type, name, option, label, cmap, color]
        variable_type, name, option, label, cmap, color = ['ws', f'{key}{cloud_type(switch)}', f'{key}{cloud_type(switch)}', r'weather state freq. [Nb day$^-1$]', 'Blues', 'b'] if key == 'ws'  else [variable_type, name, option, label, cmap, color]

        # -------------------
        # Moist static energy
        # -------------------
        variable_type, name, option, label, cmap, color = [key,  key, key, 'spec. humiid. [%]', 'Greens', 'g'] if key in ['hus'] else [variable_type, name, option, label, cmap, color]
    
    
    
    # ---------------------------
    #  Exceptions / Manually set
    # ---------------------------
    cmap = 'Reds'                                                      if switch['descent'] and switch['wap'] else cmap
    cmap = 'Blues'                                                     if switch['ascent']  and switch['wap'] else cmap
    for key in keys: # loop over true keys
        cmap = 'RdBu_r'                                                    if key == 'change with warming' else cmap
        label = '{}{}{}'.format(label[:-1], r'K$^{-1}$', label[-1:])       if key == 'per kelvin' and not label == 'ECS [K]' else label

    # cmap, color = 'Reds', 'r'
    # label = 'clim stability [K]' if  variable_type == 'stability' else None
    return metric_class(variable_type, name, option, cmap, label, color)




# ----------------------------------------------------------------------------------- variable specs (for animations) ----------------------------------------------------------------------------------------------------- #
class variable_class():
    ''' Gives variable details (name, option, label, cmap) (Used for animation of fields) '''
    def __init__(self, ref, variable_type, name, cmap, label):
        self.ref   = ref
        self.variable_type = variable_type
        self.name   = name
        self.label  = label
        self.cmap   = cmap

def get_variable_object(switch):
    ''' list of variable: name (of saved dataset), option (data array in dataset), label, cmap, color. Used for animation of fields '''
    ref, variable_type, name, label, cmap = [None, None, None, None, None]
    keys = [k for k, v in switch.items() if v]  # list of True keys
    for key in keys: # loop over true keys
        ref, variable_type, name, label, cmap = [key, 'pr', 'pr',   r'pr [mm day${^-1}$]',  'Greys']    if key in ['obj']  else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, key,  key,    r'pr [mm day${^-1}$]',  'Blues']    if key in ['pr']   else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, 'pr', 'pr',   r'pr [mm day${^-1}$]',  'Reds']     if key in ['pr99'] else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, key,  key,    'rel. humiid. [%]',     'Greens']   if key in ['hur']  else [ref, variable_type, name, label, cmap] 
        ref, variable_type, name, label, cmap = [key, 'rad', key,   r'OLR [W m${^-2}$]',    'Purples',] if key in ['rlut'] else [ref, variable_type, name, label, cmap] 
         
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



























