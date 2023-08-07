import os
import sys
home = os.path.expanduser("~")
folder_code = f'{home}/Documents/code/phd'
sys.path.insert(0, f'{folder_code}/util')
import constructed_fields as cF # imports fields for testing

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt


def find_percentile(da, percentile):
    ''' Spatial percentile of the scene '''
    return da.quantile(percentile, dim=('lat', 'lon'), keep_attrs=True)

da = cF.var2d

percentile = 0.97
aWeights = np.cos(np.deg2rad(da.lat))
pr_fixed_prRate = da.where(da >= find_percentile(da, percentile).mean(dim='time'))
pr_fixed_area = da.where(da >= find_percentile(da, percentile))


show_scene_comparison = False
if show_scene_comparison:
    fig, axes = plt.subplots(2, figsize = (8, 6))
    day = 0

    ax = axes[0]
    pr_fixed_prRate.isel(time=day).plot(ax = axes[0])
    Nb = xr.where(pr_fixed_prRate>0, 1, 0).sum(dim = ('lat', 'lon'))
    areafraction = Nb.isel(time=0).data/(len(da.lat)*len(da.lon))
    ax.set_title(f'areafraction: {round(areafraction, 4)}')
    ax.set_xlabel('')
    ax.set_xticklabels('')

    ax = axes[1]
    pr_fixed_area.isel(time=1).plot(ax = axes[1])
    Nb = xr.where(pr_fixed_area>0, 1, 0).sum(dim = ('lat', 'lon'))
    areafraction = Nb.isel(time=0).data/(len(da.lat)*len(da.lon))
    ax.set_title(f'areafraction: {round(areafraction, 4)}')

    ax.set_xlabel('')
    plt.show()



fig, ax = plt.subplots(1, figsize = (10,5))
Nb = xr.where(pr_fixed_prRate>0, 1, 0).sum(dim = ('lat', 'lon'))
areafraction = Nb/(len(da.lat)*len(da.lon))
areafraction.plot(label='fixed precipitation rate')

Nb = xr.where(pr_fixed_area>0, 1, 0).sum(dim = ('lat', 'lon'))
areafraction = Nb/(len(da.lat)*len(da.lon))
ax.axhline(areafraction.mean(dim='time'), color='r',  label='fixed area')

ax.legend(loc='upper right')
plt.title('Threshold to pick out 3% of the area')
plt.show()



