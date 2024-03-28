''' 
# --------------------------------
#   MSE - Moist Static Energy
# --------------------------------
Calculates MSE
Function:
    h = get_mse(switch, var_name, dataset, experiment, resolution, timescale)

Input:
    c_p - specific heat capacity            dim: ()                 get from: util-data/dimensions_data.py
    ta  - air temperature                   dim: (time, lat, lon)   get from: util-data/variable_base.py
    zg  - geopotential                      dim: (time, lat, lon)   get from: util-data/variable_base.py
    L_v - latent heat of vaporization       dim: ()                 get from: util-data/dimensions_data.py
    hus - specific humidity                 dim: (time, lat, lon)   get from: util-data/variable_base.py

Output:
    h: - list                               dim: (time)
'''


# ---------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------- imported scripts ---------------------------------------------------------------------------------------------- #
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import variable_base    as vB
import dimensions_data  as dD



# ------------------------
#   Calculate variable
# ------------------------
def get_mse(switch, var_name, dataset, experiment, resolution, timescale):
    ''' h = c_p * ta + zg + L_v * hus
        Where
            c_p - specific heat capacity
            ta  - air temperature
            zg  - geopotential
            L_v - latent heat of vaporization
            hus - specific humidity '''
    c_p, L_v = dD.dims_class.c_p, dD.dims_class.L_v
    ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale).load() for var in ['ta', 'zg', 'hus']]
    da = c_p * ta + zg + L_v * hus
    return da



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':
    import xarray as xr
    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print(f'running {script_name}')

    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets      as cD  

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import missing_data         as mD

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot        as mP
    import get_plot.show_plots      as sP


    script_name = os.path.splitext(os.path.basename(__file__))[0]
    switch = {
        'test_sample':          False,                                                  # data to use (test_sample uses first file (usually first year))
        'from_scratch_calc':    True,  're_process_calc':  False,                       # if both are true the scratch file is replaced with the reprocessed version (only matters for calculated variables / masked variables)
        'from_scratch':         True,  're_process':       False,                       # same as above, but for base variables
        }

    switch_test = {
        'remove_test_plots':            True,          # remove all plots in zome_plots
        'remove_script_plots':          False,          # remove plots generated from this script
        'plot_ta':                      True,          # plot 1
        'plot_zg':                      True,          # plot 2
        'plot_hus':                     True,          # plot 3
        'calc_mse':                     True,          # calculate
        'plot_mse':                     True,          # plot 4
        }
    
    sP.remove_test_plots()                                                      if switch_test['remove_test_plots']     else None
    sP.remove_folder_plots(folder = f'{os.getcwd()}/zome_plots/{script_name}')  if switch_test['remove_script_plots']   else None   

    experiment = cD.experiments[0]
    ds_ta = xr.Dataset()
    ds_zg = xr.Dataset()
    ds_hus = xr.Dataset()
    ds_mse = xr.Dataset()
    for dataset in mD.run_dataset_only(var = 'pr', datasets = cD.datasets):
        print(f'\t dataset: {dataset}')
        # ----------------------------------------------------------------------------------- Get data -------------------------------------------------------------------------------------------------- #
        ta = vB.load_variable(switch_var = {'ta': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
        zg = vB.load_variable(switch_var = {'zg': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
        hus = vB.load_variable(switch_var = {'hus': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
        ta.load()
        zg.load()
        hus.load()
        dim = dD.dims_class(ta)

        # print(ta)
        # print(zg)
        # print(hus)
        # exit()


        # ----------------------------------------------------------------------------------- Calculate --------------------------------------------------------------------------------------------------- #     
        level = 500e2
        if switch_test['plot_ta']:
            ds_ta[dataset] = ta.isel(time = 0).sel(plev = level)

        if switch_test['plot_zg']:
            ds_zg[dataset] = zg.isel(time = 0).sel(plev = level)

        if switch_test['plot_hus']:
            ds_hus[dataset] = hus.isel(time = 0).sel(plev = level)

        if switch_test['calc_mse']:
            ds_mse[dataset] = get_mse(switch, var_name = 'h', dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0]).sel(plev = level)


    # ------------------------------------------------------------------------------------------- Plot -------------------------------------------------------------------------------------------------- #
    if switch_test['plot_ta']:
        ds = ds_ta
        label = f'ta'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'a_ta_{int(level)}pa.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

    if switch_test['plot_zg']:
        ds = ds_zg
        label = f'zg'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'b_zg_{int(level)}pa.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

    if switch_test['plot_hus']:
        ds = ds_hus
        label = f'hus'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'c_hus_{int(level)}pa.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')

    if switch_test['plot_mse']:
        ds = ds_hus
        label = f'mse'
        vmin = None
        vmax = None
        cmap = 'Blues'
        filename = f'd_mse_{int(level)}pa.png'
        fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
        sP.show_plot(fig, show_type = 'save_cwd', filename = f'{script_name}/{filename}')






