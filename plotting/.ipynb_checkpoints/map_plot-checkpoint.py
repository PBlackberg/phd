#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit

from scipy.io import loadmat
#%%

# 
#
# Map plot (single)
#
#

# numpy files
# csv or text for names 


# directories
data_path = '/c/Users/carlp/data/'
project = 'cmip5'
percentile= '97th_percentile'
the_type= ['domain', 'convective_objects', 'aggregation_index', 'examples']

model_inst = ['IPSL','NOAA-GFDL','NASA-GISS','BCC','CNRM-CERFACS','NCAR'
        ,'NIMR-KMA','BNU','ICHEC','LASG-CESS','MPI-M','CMCC','INM','NCC' 
        ,'CCCma','MIROC','MOHC','MRI','NSF-DOE-NCAR']
var= ['pr','hus','tas','convective_objects','aggregation_index']
scenario= ['historical', 'rcp85']


path = data_path +'/'+ percentile +'/'+ the_type[0]   
fileName= model_inst[0] +'/'+ the_type[0] +'/'+  var[0] +'/'+ scenario[0]
dir= path +'/'+ fileName


# read.csv('path','r')
# # np.genfromtxt('path',delimiter=',', skip_header=1)
# pd.dataframe()
# xarray (netcdf)


# ask holger about terminal environment
# conda environment gadi (NCI)


#%%
# loading variables
data = loadmat(r"C:\Users\carlp\data\97th_percentile\domain\IPSL_domain_pr_historical.mat")
# data.keys() to see the variables
x = np.squeeze(data['lon'],axis=1)
y = np.squeeze(data['lat'],axis=1)
z = data['snapshot_pr_image']
z1 = np.transpose(z)

#%%

# plot figure
plt.figure(figsize=(8,5))
plt.pcolor(x,y,z1)

#








# %%
