
## folder and filename for leading instead of path
# folder = metric.get_metric_folder(mV.folder_save[0], metric.name, source)
# filename = metric.get_filename(metric.name, source, dataset, 'daily', mV.experiments[0], mV.resolutions[0]) 
# filename = metric.get_filename(metric.name, source, 'GPCP', 'daily', mV.experiments[0], mV.resolutions[0]) if dataset in ['ERA5', 'CERES'] else filename 

# folder = metric.get_metric_folder(mV.folder_save[0], f'{metric.name}_snapshot', source)
# filename = metric.get_filename(f'{metric.name}_snapshot', source, dataset, mV.timescales[0], mV.experiments[0], mV.resolutions[0])



a = 5


def func(a):
    b = True
    a = 6 if b else a

func(a)
print(a)



