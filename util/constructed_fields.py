import xarray as xr
import numpy as np
import pandas as pd

# ------------------------------------------------------------------------------------- For 2D variables, like surface pr ----------------------------------------------------------------------------------------------------- #
var2d = np.random.rand(10961, 22, 128)
time_range = pd.date_range("1970/01/01","2000/01/05",freq='D',inclusive='left')
lat = np.linspace(-30, 30, 22)
lon = np.linspace(0, 360, 128)
var2d = xr.DataArray(
    data = var2d,
    dims=['time','lat','lon'],
    coords={'time': time_range, 'lat': lat, 'lon': lon}
    )

# --------------------------------------------------------------------------- For 3D variables, like vertical pressure velocity (wap) ----------------------------------------------------------------------------------------------------- #
var3d = np.random.rand(4, 22, 128)
time_range = pd.date_range("1970/01/01","1970/01/05",freq='D',inclusive='left')
lat = np.linspace(-30, 30, 22)
lon = np.linspace(0, 360, 128)
var3d = xr.DataArray(
    data = var3d,
    dims=['time','lat','lon'],
    coords={'time': time_range,'lat': lat,'lon': lon}
    )

# -------------------------------------------------------------------- Scene with contiguous convective regions (for testing organization metrics) ----------------------------------------------------------------------------------------------------- #
orgScenes = np.zeros(shape = (4, 22, 128))
time_range = pd.date_range("1970/01/01","1970/01/05",freq='D',inclusive='left')
lat = np.linspace(-30, 30, 22)
lon = np.linspace(0, 360, 128)
orgScenes = xr.DataArray(
    data = orgScenes,
    dims=['time','lat','lon'],
    coords={'time': time_range,'lat': lat,'lon': lon}
    )

# one object
orgScenes[0, :, :] = 0
orgScenes[0, 10:15, 60:70] = 2

# two objects (across boundary)
orgScenes[1, :, :] = 0
orgScenes[1, 10:15, 60:70] = 2

orgScenes[1, 5:8, 0:10] = 2
orgScenes[1, 5:10, 125:] = 2

# two objects (do not cross boundary, but distance between objects across boundary is closer)
orgScenes[2, :, :] = 0
orgScenes[2, 5:8, 2:10] = 2

orgScenes[2, 10:15, 120:-5] = 2

# multiple objects (including crossing boundary multiple times) (9 objects)
orgScenes[3, :, :] = 0
orgScenes[3, 10:15, 60:70] = 2

orgScenes[3, 5:8, 0:10] = 2
orgScenes[3, 5:10, 125:] = 2

orgScenes[3, 17:20, 0:3] = 2
orgScenes[3, 17:19, 125:] = 2

orgScenes[3, 16:18, 15:19] = 2

orgScenes[3, 3:5, 30:40] = 2

orgScenes[3, 10:17, 92:95] = 2

orgScenes[3, 6:7, 105:106] = 2

orgScenes[3, 2:4, 80:85] = 2

orgScenes[3, 18:20, 35:39] = 2
















