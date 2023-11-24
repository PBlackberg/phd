
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt


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
print(ds)



plt.figure()
ds.sst_anom.plot()

plt.show()




