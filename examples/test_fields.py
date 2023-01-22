import xarray as xr
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



artScenes_4days = np.zeros(shape = (4, 22, 128))
time_range = pd.date_range("1970/01/01","1970/01/05",freq='D',inclusive='left')
lat = np.linspace(-30, 30, 22)
lon = np.linspace(0, 360, 128)

artScenes_4days = xr.DataArray(
                    data=artScenes_4days,
                    dims=['time','lat', 'lon'],
                    coords={'time': time_range, 'lat': lat, 'lon': lon}
)


# one object
artScenes_4days[0, :, :] = 0
artScenes_4days[0, 10:15, 60:70] = 2



# two objects (across boundary)
artScenes_4days[1, :, :] = 0
artScenes_4days[1, 10:15, 60:70] = 2

artScenes_4days[1, 5:8, 0:10] = 2
artScenes_4days[1, 5:10, 125:] = 2



# two objects (do not cross boundary, but distance between objects across boundary is closer)
artScenes_4days[2, :, :] = 0
artScenes_4days[2, 5:8, 2:10] = 2

artScenes_4days[2, 10:15, 120:-5] = 2



# multiple objects (including crossing boundary multiple times) (9 objects)
artScenes_4days[3, :, :] = 0
artScenes_4days[3, 10:15, 60:70] = 2

artScenes_4days[3, 5:8, 0:10] = 2
artScenes_4days[3, 5:10, 125:] = 2

artScenes_4days[3, 17:20, 0:3] = 2
artScenes_4days[3, 17:19, 125:] = 2

artScenes_4days[3, 16:18, 15:19] = 2

artScenes_4days[3, 3:5, 30:40] = 2

artScenes_4days[3, 10:17, 92:95] = 2

artScenes_4days[3, 6:7, 105:106] = 2

artScenes_4days[3, 2:4, 80:85] = 2

artScenes_4days[3, 18:20, 35:39] = 2



















