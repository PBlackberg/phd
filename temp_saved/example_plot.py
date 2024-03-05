'''
# ----------------
#   pe_obj_plot
# ----------------
Visualizing precipitation efficiency in convective objects (based on precipitation threshold)
'''


# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np
import xarray as xr


# ------------------------------------------------------------------------------------- imported scripts ---------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD  

sys.path.insert(0, f'{os.getcwd()}/util-data')
import missing_data as mD
import pe_var as pE
import conv_obj_var as cO

sys.path.insert(0, f'{os.getcwd()}/util-calc')
import ls_state.means_calc as mC

sys.path.insert(0, f'{os.getcwd()}/util-plot')
import get_plot.map_plot as mP


switch_test = {
    'delete_previous_plots':        True,
    'plot_pe':                      False,
    'plot_conv':                    False,
    'plot_pe_conv':                 True,
    }

mP.remove_test_plots() if switch_test['delete_previous_plots'] else None
experiment = cD.experiments[0]

ds_pe = xr.Dataset()
ds_conv = xr.Dataset()
ds_pe_conv = xr.Dataset()
for dataset in mD.run_dataset_only(var = 'pe', datasets = cD.datasets):
    # -------------------------------------------------------------------------------------- Get data --------------------------------------------------------------------------------------------------- #
    switch = {'ocean_mask': False}
    pe = pE.get_pe(switch = switch, var_name = 'pe', dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = 'monthly')  
    # print(pe)

    switch = {'fixed_area': False}
    conv = cO.get_conv_var(switch, dataset, experiment)
    conv_obj, obj_id = cO.get_conv_obj(switch, dataset, experiment)
    # print(conv)
    # print(conv_obj)
    # print(obj_id)

    pe.load()
    conv.load()
    conv_obj.load()
    obj_id.load()


    # ------------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_pe']:
        ds_pe[dataset] = pe.isel(time = 0)

    if switch_test['plot_conv']:
        ds_conv[dataset] = conv.isel(time = 0)

    if switch_test['plot_pe_conv']:
        ds_pe_conv[dataset] = pe.isel(time = 0).where(conv.isel(time = 0) > 0)


    # ---------------------------------------------------------------------------------------- plot --------------------------------------------------------------------------------------------------- #
    if switch_test['plot_pe']:
        label = 'kg'
        vmin = 0
        vmax = 250
        cmap = 'Blues'
        title = f'Precipitaiton_efficiency_{dataset}'
        filename = f'{title}.png'
        fig, ax = mP.plot_dsScenes(ds_pe, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds_pe.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_conv']:
        label = 'conv [0,1]'
        vmin = 0
        vmax = 1
        cmap = 'Greys'
        title = f'convective_regions_{dataset}'
        filename = f'{title}.png'
        fig, ax = mP.plot_dsScenes(ds_conv, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds_conv.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)

    if switch_test['plot_pe_conv']:
        label = 'kg'
        vmin = 0
        vmax = 250
        cmap = 'Blues'
        title = f'Precipitaiton_efficiency_in_objects_{dataset}'
        filename = f'{title}.png'
        fig, ax = mP.plot_dsScenes(ds_pe_conv, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds_pe_conv.data_vars.keys()))
        mP.show_plot(fig, show_type = 'save_cwd', filename = filename)










































