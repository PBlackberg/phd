
import xarray as xr
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
home = os.path.expanduser("~")                           
sys.path.insert(0, f'{os.getcwd()}/switch')
import myVars as mV                                 
import myFuncs as mF     
import myClasses as mC


a = xr.open_dataset('sst.mnmean.nc')
print(a)





NOAA_tas_monthly__regridded.nc



