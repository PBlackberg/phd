import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(0)  # For reproducibility
data = np.random.choice([0, 1], size=(30, 10, 10))
time = pd.date_range("2023-01-01", periods=30)
xarray_data = xr.DataArray(data, dims=["time", "x", "y"], coords={"time": time})
heatmap_data = xarray_data.sum(dim="time")


plot = False
if plot:
    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap_data, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Frequency of Objects')
    plt.title('Heatmap of Object Frequency Over Time')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()




# Loop through days and create the objects. Then add the scenes and create a heatmap that way from each scenario.
# This can be a metric calculated from org_met.py











