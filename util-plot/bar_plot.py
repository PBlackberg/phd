''' 
# ----------------
#   Bar plot
# ----------------
This script plots a bar plot of each model's resolution 
'''

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core/myFuncs_plots')
import myFuncs_plots        as mFp


def plot_dsBar(ds, dim = 'model', title = 'test', ylabel = '', highlight_given = False, models_highlight = ['a', 'b']):
    datasets, alist = [], []
    for dataset in ds.data_vars:
        datasets.append(dataset)
        alist.append(ds[dataset])

    sorted_indices = np.argsort(alist)
    sorted_datasets = [datasets[i] for i in sorted_indices]
    sorted_alist = [alist[i] for i in sorted_indices]

    da = xr.DataArray(data = sorted_alist, dims = [dim], coords={dim: sorted_datasets})
    fig = plt.figure(figsize = (10, 5))
    ax = fig.add_subplot()

    if highlight_given:
        # highlight_color = 'blue'
        highlight_color = 'green'
    else:
        highlight_color = 'C0'  # This is orange, but you can use any color you like

    default_color = 'k'  # This is blue, the default color matplotlib uses
    if highlight_given:
        colors = [highlight_color if dataset in models_highlight else default_color for dataset in sorted_datasets]
    else:
        colors = [highlight_color if i < 14 else default_color for i in range(len(sorted_datasets))]

    da.to_series().plot.bar(ax=ax, color = colors, rot=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    mFp.scale_ax_y(ax, scaleby=0.8)
    mFp.move_row(ax, moveby = 0.1)
    mFp.move_row(ax, moveby = 0.065)

    if highlight_given:
        label = 'pwad and hur close to obs'
    else:
        label='pwad close to obs'

    close_patch = mpatches.Patch(color=highlight_color, label=label)
    plt.legend(handles=[close_patch])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return fig, ax








# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import numpy as np
    import os
    import sys
    home = os.path.expanduser("~")
    sys.path.insert(0, f'{os.getcwd()}/switch')
    import myVars as mV
    import myFuncs as mF


    switch = {'show': False, 'save_to_desktop': True}
    # --------------------------------------------------------------------------------------- Get resolution ----------------------------------------------------------------------------------------------------- #
    dlats, dlons, res, model_list = [], [], [], []
    for model in mV.datasets:
        # print(model)
        source = mV.find_source(model, mV.models_cmip5, mV.models_cmip6, mV.observations)
        da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/pr/{source}/{model}_pr_{mV.timescales[0]}_{mV.experiments[0]}_orig.nc')['pr']
        dlat = da['lat'][2] - da['lat'][1]
        dlon = da['lon'][2] - da['lon'][1]
        
        dlats = np.append(dlats, dlat)
        dlons = np.append(dlons, dlon)
        res = np.append(res, dlat * dlon)
        model_list = np.append(model_list, model)
    dlats = xr.DataArray(data = dlats, dims = ['model'], coords={'model': model_list})
    dlons = xr.DataArray(data = dlons, dims = ['model'], coords={'model': model_list})
    res = xr.DataArray(data = res, dims = ['model'], coords={'model': model_list})


    # ------------------------------------------------------------------------------------------- bar plot ----------------------------------------------------------------------------------------------------- #
    plot_lon = False
    if plot_lon:
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot()
        dlons.to_series().plot.bar(ax=ax, rot=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.title('dlon from cmip6 models')
        plt.ylabel('dlon')
        mF.scale_ax_y(ax, scaleby=0.8)
        mF.move_row(ax, moveby = 0.1)
        mF.move_row(ax, moveby = 0.065)

        plt.show() if switch['show'] else None
        mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save_to_desktop'] else None
        print('finished')

    plot_lat = False
    if plot_lat:
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot()
        dlats.to_series().plot.bar(ax=ax, rot=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.title('dlat from cmip6 models')
        plt.ylabel('dlat')
        mF.scale_ax_y(ax, scaleby=0.8)
        mF.move_row(ax, moveby = 0.1)
        mF.move_row(ax, moveby = 0.065)

        plt.show() if switch['show'] else None
        mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save_to_desktop'] else None
        print('finished')

    plot_res = True
    if plot_res:
        fig = plt.figure(figsize = (10, 5))
        ax = fig.add_subplot()
        res.to_series().plot.bar(ax=ax, rot=0)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        plt.title('Resolution of cmip6 models')
        plt.ylabel(r'dlat x dlon [$\degree$]')
        mF.scale_ax_y(ax, scaleby=0.8)
        mF.move_row(ax, moveby = 0.1)
        mF.move_row(ax, moveby = 0.065)

        plt.axhline(y=2.5, color='k', linestyle='--')

        plt.show() if switch['show'] else None
        mF.save_figure(fig, f'{home}/Desktop', 'test.pdf') if switch['save_to_desktop'] else None
        print('finished')

