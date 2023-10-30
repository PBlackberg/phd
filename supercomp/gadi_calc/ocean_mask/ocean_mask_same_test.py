import numpy as np
import xarray as xr
import os
import sys
home = os.path.expanduser("~")
sys.path.insert(0, f'{os.getcwd()}/util')
import constructed_fields as cF
import get_data as gD
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV
import myClasses as mC
import myFuncs as mF
import matplotlib.pyplot as plt



import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


print('started')
# ---------------------------
# get variable and ocean mask
# ---------------------------
# --------------------------------------------------------------------------------- variable (pr) ----------------------------------------------------------------------------------------------------------#
folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/day/atmos/day/r1i1p1/v20161204/pr'
filename = 'pr_day_FGOALS-g2_historical_r1i1p1_19750101-19751231.nc'
ds_pr = xr.open_dataset(f'{folder}/{filename}').isel(time=0).sel(lat = slice(-35, 35))
pr = ds_pr['pr']

scene = pr
fig, ax = mF.create_map_figure(width = 12, height = 4)
pcm = mF.plot_axMapScene(ax, scene, 'Blues', vmin = None, vmax = None)
mF.move_col(ax, moveby = -0.055)
mF.move_row(ax, moveby = 0.075)
mF.scale_ax(ax, scaleby = 1.15)
mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = 'pr', text_pad = 0.125)
mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
mF.plot_axtitle(fig, ax, 'ocean mask', xpad = 0.005, ypad = 0.025, fontsize = 15)
mF.format_ticks(ax, labelsize = 11)
switch = {'save_folder_cwd': True}
mF.save_plot(switch, fig, home, 'test_pr')


# ------------------------------------------------------------------------------------ ocean mask ----------------------------------------------------------------------------------------------------------#
folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/fx/ocean/fx/r0i0p0/v20161204/sftof'
filename = 'sftof_fx_FGOALS-g2_historical_r0i0p0.nc'
ds = xr.open_dataset(f'{folder}/{filename}').sel(lat = slice(-35, 35))
da = ds['sftof']

# scene = da
# fig, ax = mF.create_map_figure(width = 12, height = 4)
# pcm = mF.plot_axMapScene(ax, scene, 'Blues', vmin = None, vmax = None)
# mF.move_col(ax, moveby = -0.055)
# mF.move_row(ax, moveby = 0.075)
# mF.scale_ax(ax, scaleby = 1.15)
# mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = 'pr', text_pad = 0.125)
# mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
# mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
# mF.plot_axtitle(fig, ax, 'ocean mask', xpad = 0.005, ypad = 0.025, fontsize = 15)
# mF.format_ticks(ax, labelsize = 11)
# print(da)
# print(fig)
# switch = {'save_folder_cwd': True}
# mF.save_plot(switch, fig, home, 'ocean_mask')



# ---------------------------
#     Apply ocean mask
# ---------------------------
# ------------------------------------------------------------------------------ Regred dataset and ocean mask ----------------------------------------------------------------------------------------------------------#
# print(pr)
# print(da)
import regrid_xesmf as regrid
regridder = regrid.regrid_conserv_xesmf(ds_pr)          # define regridder based of grid from other model
pr = regridder(pr)                                      # conservatively interpolate data onto grid from other model
regridder = regrid.regrid_conserv_xesmf(ds)             # define regridder based of grid from other model
da = regridder(da)/100                                  # conservatively interpolate data onto grid from other model
da = da.where(da == 1)                                  # only keep binary mask (fractional values may not physically represent what the fraction of ocean mean) 
# print(pr)
# print(da)

ds = xr.Dataset(data_vars = {'ocean_mask': da.sel(lat=slice(-30,30))}, attrs = ds.attrs) 
mF.save_file(ds, folder='/home/565/cb4968/Documents/code/phd', filename='ocean_mask.nc')

scene = da
fig, ax = mF.create_map_figure(width = 12, height = 4)
pcm = mF.plot_axMapScene(ax, scene, 'Blues', vmin = None, vmax = None)
mF.move_col(ax, moveby = -0.055)
mF.move_row(ax, moveby = 0.075)
mF.scale_ax(ax, scaleby = 1.15)
mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = 'pr', text_pad = 0.125)
mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
mF.plot_axtitle(fig, ax, 'ocean mask', xpad = 0.005, ypad = 0.025, fontsize = 15)
mF.format_ticks(ax, labelsize = 11)
print(da)
switch = {'save_folder_cwd': True}
mF.save_plot(switch, fig, home, 'ocean_mask')






# ------------------------------------------------------------------------------------ mask variable ----------------------------------------------------------------------------------------------------------#
scene = pr * da
fig, ax = mF.create_map_figure(width = 12, height = 4)
pcm = mF.plot_axMapScene(ax, scene, 'Blues', vmin = None, vmax = None)
mF.move_col(ax, moveby = -0.055)
mF.move_row(ax, moveby = 0.075)
mF.scale_ax(ax, scaleby = 1.15)
mF.cbar_below_axis(fig, ax, pcm, cbar_height = 0.05, pad = 0.15, numbersize = 12, cbar_label = 'pr', text_pad = 0.125)
mF.plot_xlabel(fig, ax, 'Lon', pad = 0.1, fontsize = 12)
mF.plot_ylabel(fig, ax, 'Lat', pad = 0.055, fontsize = 12)
mF.plot_axtitle(fig, ax, 'ocean mask', xpad = 0.005, ypad = 0.025, fontsize = 15)
mF.format_ticks(ax, labelsize = 11)
switch = {'save_folder_cwd': True}
mF.save_plot(switch, fig, home, 'test_ocean_mask')


print('finished')









# individual ocean masks
    # if switch['ocean_mask']:
    #     ensemble = choose_cmip6_ensemble(model, 'historical')

    #     if model in ['ACCESS-ESM1-5', 'ACCESS-CM2']:
    #         path_gen = f'/g/data/fs38/publications/CMIP6/CMIP/{mV.institutes[model]}/{model}/historical/{ensemble}/fx/sftlf/{folder_grid}'
    #         version = 'latest'
    #     else:            
    #         path_gen = f'/g/data/oi10/replicas/CMIP6/CMIP/{mV.institutes[model]}/{model}/historical/{ensemble}/fx/sftlf/{folder_grid}'
    #         version = latestVersion(path_gen)
        
    #     folder = f'{path_gen}/{version}'
    #     filename = f'sftlf_fx_{model}_historical_{ensemble}_{folder_grid}.nc'
    #     print(f'{folder}/{filename}')
    #     mask = ((xr.open_dataset(f'{folder}/{filename}')['sftlf'].sel(lat=slice(-35,35))/100)-1)*(-1)
    #     if model in ['MPI-ESM1-2-LR']: # the coordinates misalign by e-14
    #         da['lat'] = da['lat'].round(decimals=6)
    #         mask['lat'] = mask['lat'].round(decimals=6)

    #     if model in ['ACCESS-ESM1-5']: # coordinates mosaligned
    #         mask = mask.interp(lat=da['lat'])
    #         mask['lon'] = mask['lon'] + 0.9375 # comment out, and just use mask from lasg-cess



# implementation of ocean mask

            #     folder = '/g/data/al33/replicas/CMIP5/combined/LASG-CESS/FGOALS-g2/historical/fx/ocean/fx/r0i0p0/v20161204/sftof'
            # filename = 'sftof_fx_FGOALS-g2_historical_r0i0p0.nc'
            # ds_in = xr.open_dataset(f'{folder}/{filename}').sel(lat = slice(-35, 35))
            # mask = ds_in['sftof']
            # regridder = regrid.regrid_conserv_xesmf(ds_in)  # define regridder based of grid from other model
            # mask = regridder(mask)                          # conservatively interpolate data onto grid from other model (that model's ocean maks and data not on same grid)
            