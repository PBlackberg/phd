import numpy as np
from scipy.interpolate import interp1d
import xarray as xr

x = [100, 90, 80, 70, 60, 50]
y = [10, 8, 20, 10, 30, 20]
x_interp = [95, 75, 65]

data = xr.DataArray(y, coords={'x': x}, dims='x')

# period = 100                                                                                        # Set period to the maximum pressure value
# y_interp = np.interp(x_interp, x, y, period=period)                                                 # This is doing it correctly

# y_interp = np.interp(x_interp, x, y)                                                                # This is not doing it correctly

# f = interp1d(x, y, kind='linear', fill_value='extrapolate')                                         # This is doing it corretly too
# y_interp = f(x_interp)

import warnings                                                                                       # This works as well
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
y_interp = data.interp(x=x_interp, method='linear', kwargs={"fill_value": "extrapolate"}).data        
warnings.resetwarnings()

for xi, yi in zip(x_interp, y_interp):
    print(f"Interpolated value at x={xi}: y={yi}")


