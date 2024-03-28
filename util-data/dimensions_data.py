'''
# ------------------------
#    Dimensions data
# ------------------------
Dimensions of spatial data variables, and commonly used constants

Call from different script with:
import os
import sys
sys.path.insert(0, f'{os.getcwd()}/util-data')
import dimensions_data as dD
'''


# --------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------- #
import numpy as np



class dims_class():
    R = 6371        # radius of earth
    g = 9.81        # gravitaional constant
    c_p = 1.005     # specific heat capacity
    L_v = 2.256e6   # latent heat of vaporization
    def __init__(self, da):
        self.lat, self.lon       = da['lat'].data, da['lon'].data
        self.lonm, self.latm     = np.meshgrid(self.lon, self.lat)
        self.dlat, self.dlon     = da['lat'].diff(dim='lat').data[0], da['lon'].diff(dim='lon').data[0]
        self.aream               = np.cos(np.deg2rad(self.latm))*np.float64(self.dlon*self.dlat*self.R**2*(np.pi/180)**2) # area of domain: cos(lat) * (dlon*dlat) (area of gridbox decrease towards the pole as gridlines converge)
        self.latm3d, self.lonm3d = np.expand_dims(self.latm,axis=2), np.expand_dims(self.lonm,axis=2)                     # used for broadcasting
        self.aream3d             = np.expand_dims(self.aream,axis=2)



# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':

    import xarray as xr

    import os
    import sys
    sys.path.insert(0, f'{os.getcwd()}/util-core')
    import choose_datasets as cD

    sys.path.insert(0, f'{os.getcwd()}/util-data')
    import variable_calc as vC
    dataset = cD.datasets[0]
    experiment = cD.experiments[0]

    sys.path.insert(0, f'{os.getcwd()}/util-plot')
    import get_plot.map_plot    as mP

    switch = {
        'test_sample':  False, 
        'fixed_area':   False,
        'from_scratch': True,   're_process':   False,  # if taking data from scratch or calculating
        }

    switch_test = {
        'delete_previous_plots':    True,
        }
    mP.remove_test_plots() if switch_test['delete_previous_plots'] else None


    # ---------------------------------------------------------------------------------------- Get data ----------------------------------------------------------------------------------------------------- #
    da, _ = vC.get_variable(switch_var = {'pr': True}, switch = switch, dataset = dataset, experiment = experiment, resolution = cD.resolutions[0], timescale = cD.timescales[0])
    dim = dims_class(da)
    # print(dim)
    # print(f'Radius of Earth: {dim.R}')
    # print(f'Area matrix: {dim.aream}')
    # print(f'Area matrix shape: {np.shape(dim.aream)}')


    # ------------------------------------------------------------------------------------------ Plot ----------------------------------------------------------------------------------------------------- #
    ds = xr.Dataset()
    ds['area_matrix'] = xr.DataArray(dim.aream, dims = ['lat', 'lon'], coords = {'lat': dim.lat, 'lon': dim.lon})
    cmap = 'Blues'
    vmin = None
    vmax = None
    label = 'area [km^2]'
    filename = 'area_matrix'
    fig, ax = mP.plot_dsScenes(ds, label = label, title = filename, vmin = vmin, vmax = vmax, cmap = cmap, variable_list = list(ds.data_vars.keys()))
    mP.show_plot(fig, show_type = 'save_cwd', filename = filename)













































