
import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
from scipy.signal import detrend
from eofs.xarray import Eof
import pandas as pd

# used data from here: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.html
# url = 'http://www.esrl.noaa.gov/psd/thredds/dodsC/Datasets/noaa.ersst.v5/sst.mnmean.nc'
# ds = xr.open_dataset(url)

ds = xr.open_dataset('/Users/cbla0002/Documents/data/sample_data/tas/obs/sst.mnmean.nc')
ds = ds.sel(time=slice('1998', '2022'))
# print(ds)
ds.load()
sst_clim = ds.sst.groupby('time.month').mean(dim='time')
sst_anom = ds.sst.groupby('time.month') - sst_clim
sst_anom_detrended = xr.apply_ufunc(detrend, sst_anom.fillna(0), kwargs={'axis': 0}).where(~sst_anom.isnull())

weights = np.cos(np.deg2rad(ds.lat)).where(~sst_anom[0].isnull())
weights /= weights.mean()


plot = False
if plot:
    (sst_anom * weights).mean(dim=['lon', 'lat']).plot(label='raw')
    (sst_anom_detrended * weights).mean(dim=['lon', 'lat']).plot(label='detrended SST')
    plt.grid()
    plt.legend()
    plt.show()

# nino 3.4 index (ONI, calc)
sst_anom_nino34 = sst_anom_detrended.sel(lat=slice(5, -5), lon=slice(190, 240))
sst_anom_nino34_mean = sst_anom_nino34.mean(dim=('lon', 'lat'))
oni = sst_anom_nino34_mean.rolling(time=3, center=True).mean()
# print('oni lenght:', len(oni))

plot = False
if plot:
    oni.plot()
    plt.grid()
    plt.ylabel('Anomaly (dec. C)')
    plt.axhline(0.5, linestyle = '--')
    plt.axhline(-0.5, linestyle = '--')
    plt.show()


# eofs (the first principle component is almost identical to the nino index)
sst_anom_detrended = sst_anom_detrended.drop_vars(['lat', 'lon', 'month'])
solver = Eof(sst_anom_detrended, weights=np.sqrt(weights))
eof1 = solver.eofsAsCorrelation(neofs=1)
pc1 = solver.pcs(npcs=1, pcscaling=1)


# nino 3.4 index (ONI, from website)
df = pd.read_csv('/Users/cbla0002/Documents/data/sample_data/tas/obs/oni_index.csv', delim_whitespace=True)
# print(df)

season_to_month = {
    'DJF': '01-01', 
    'JFM': '02-01', 
    'FMA': '03-01', 
    'MAM': '04-01',
    'AMJ': '05-01', 
    'MJJ': '06-01', 
    'JJA': '07-01', 
    'JAS': '08-01', 
    'ASO': '09-01', 
    'SON': '10-01',
    'OND': '11-01',
    'NDJ': '12-01'
}

df['time'] = pd.to_datetime(df['YR'].astype(str) + '-' + df['SEAS'].map(season_to_month))
ds = df.set_index('time').to_xarray()
ds = ds.rename({'ANOM': 'sst_anom'})
ds = ds.sel(time=slice('1998', '2022'))
ds = -1*ds
# print(ds)


plot = True
if plot:
    # pc1.sel(mode=0).plot(label='PC mode 0')
    ds.sst_anom.plot(label='ONI website')
    (-oni).plot(label='- ONI calc')
    plt.axhline(0.5, linestyle = '--')
    plt.axhline(-0.5, linestyle = '--')
    plt.grid()
    plt.legend()
    plt.show()





