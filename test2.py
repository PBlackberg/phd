# def find_general_metric_and_specify_cbar(switch):
#     if switch['pr'] or switch['percentiles_pr'] or switch['rx1day_pr'] or switch['rx5day_pr']:
#         variable_type = 'pr'
#         cmap = 'Blues'
#         cbar_label = 'pr [mm day{}]'.format(mF.get_super('-1'))
#     if switch['pr']:
#         metric = 'pr' 
#         metric_option = metric
#     if switch['percentiles_pr']:
#         metric = 'percentiles_pr' 
#         metric_option = 'pr97' # there is also pr95, pr99
#     if  switch['rx1day_pr'] or switch['rx5day_pr']:
#         metric = 'rxday_pr'
#         metric_option = 'rx1day_pr' if switch['rx1day_pr'] else 'rx5day_pr'

#     if switch['wap']:
#         variable_type = 'wap'
#         cmap = 'RdBu_r' if not switch['ascent'] and not switch['descent'] else 'Reds'
#         cbar_label = 'wap [hPa day' + mF.get_super('-1') + ']'
#         region = name_region(switch)
#         metric = f'wap{region}'
#         metric_option = metric

#     if switch['tas']:
#         variable_type = 'tas'
#         cmap = 'RdBu_r'
#         cbar_label = 'Temperature [\u00B0C]'
#         region = name_region(switch)
#         metric = f'tas{region}'
#         metric_option = metric

#     if switch['hur']:
#         variable_type = 'hur'
#         cmap = 'Greens'
#         cbar_label = 'Relative humidity [%]'
#         region = name_region(switch)
#         metric = f'hur{region}'
#         metric_option = metric

#     if switch['rlut']:
#         variable_type = 'lw'
#         cmap = 'Purples'
#         cbar_label = 'OLR [W m' + mF.get_super('-2') +']'
#         region = name_region(switch)
#         metric = f'rlut{region}'
#         metric_option = metric

#     if switch['lcf'] or switch['hcf']:
#         variable_type = 'cl'
#         cmap = 'Blues'
#         cbar_label = 'cloud fraction [%]'
#         region = name_region(switch)
#         metric = f'lcf{region}' if switch['lcf'] else f'hcf{region}'
#         metric_option = metric

#     if switch['hus']:
#         variable_type = 'hus'
#         cmap = 'Greens'
#         cbar_label = 'Specific humidity [mm]'
#         region = name_region(switch)
#         metric = f'hus{region}'
#         metric_option = metric

#     if switch['change with warming']:
#         cmap = 'RdBu_r'
#         cbar_label = '{}{} K{}'.format(cbar_label[:-1], mF.get_super('-1'), cbar_label[-1:]) if switch['per_kelvin'] else cbar_label
#     return variable_type, metric, metric_option, cmap, cbar_label

# def specify_metric_and_title(switch, metric, metric_option):
#     if switch['snapshot']:
#         title = f'{metric_option} snapshot'
#         metric = f'{metric}_snapshot'
#         metric_option = f'{metric_option}_snapshot'

#     if switch['climatology'] or switch['change with warming']:
#         metric = f'{metric}_tMean'
#         metric_option = f'{metric_option}_tMean'
#         title = f'{metric_option} time mean' if switch['climatology'] else f'{metric_option}, change with warming'
#     return metric, metric_option, title



    # variable_type, metric, metric_option, cmap, cbar_label = find_general_metric_and_specify_cbar(switch)


