# ------------------------
#    data variable class
# ------------------------
# ----------------------------------------------------------------- Gives Variable specs ----------------------------------------------------------------------------------------------------- #
class variable_class():
    ''' Gives variable details (name, option, label, cmap) (Used for animation of fields) '''
    def __init__(self, ref, variable_type, name, cmap, label):
        self.ref           = ref
        self.variable_type = variable_type
        self.name          = name
        self.label         = label
        self.cmap          = cmap


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

