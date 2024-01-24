a = 'pr_rx1day'
print(a)

def fix_metric_type_name(a):
    a = f'{a}_this'
        # metric_type_name = f'{metric_type_name}_{mV.conv_percentiles[0]}thprctile'
        # metric_type_name = f'{metric_type_name}_fixed_area' if switch['fixed_area'] else metric_type_name
    return a

a = fix_metric_type_name(a)
print(a)


