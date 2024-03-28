'''
# --------------
#  Heat map
# --------------
Visualizes vertical profile across latitude values
'''


# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt














# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
# ------------------------------------------------------------------------------------ imported scripts --------------------------------------------------------------------------------------------------- #
    import os
    import sys
    home = os.path.expanduser("~")
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD           
    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import metric_data as mD


# --------------------------------------------------------------------------------------- get data --------------------------------------------------------------------------------------------------- #
    metric_type = 'conv_org'
    metric_name = f'obj_snapshot_{cD.conv_percentiles[0]}thprctile'
    ds = xr.Dataset()
    for dataset in cD.datasets:
        da = mD.load_metric(metric_type, metric_name, dataset, cD.experiments[0])
        ds[dataset] = da>0
    # print(ds)































