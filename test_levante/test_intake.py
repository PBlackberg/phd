# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import xarray as xr
import numpy as np
from pathlib import Path
from dask.utils import format_bytes # check size of variable: print(format_bytes(da.nbytes))



# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")                            
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myFuncs2 as mF2      


print('testing intake')
import intake
import pandas as pd
pd.set_option("max_colwidth", None) 
import outtake



catalog_file = "/work/ka1081/Catalogs/dyamond-nextgems.json"
cat = intake.open_esm_datastore(catalog_file)
# print(mF2.get_from_cat(cat, ["project", "experiment_id", "source_id", "simulation_id"]))

print('ICON ngc2013')
# looking at available variables
# columns = ['variable_id']
# print(cat.search(simulation_id="ngc2013", realm = 'atm', frequency = '3hour').df[columns])
# dataF = cat.search(simulation_id="ngc2013", realm = 'atm').df
# unique_freq = dataF['frequency'].unique()
# print(unique_freq)
# columns = ["realm", "variable_id"]
# print(cat.search(simulation_id="ngc2013", frequency = '3hour').df[columns])

# getting data
hits = cat.search(simulation_id="ngc2013", variable_id="pr", frequency="3hour")
dataset_dict = hits.to_dataset_dict(cdf_kwargs={"chunks": {"time": 1}})
keys = list(dataset_dict.keys())
dataset = dataset_dict[keys[0]]
print(dataset)
print(dataset.pr.isel(time=0).max().values)
print(dataset.pr.isel(time=1).max().values)













