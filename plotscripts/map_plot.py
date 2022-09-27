import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import timeit


# 
#
# Map plot (single)
#
#



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
dir= path +'/'+ file


# loading variables
file = open(,'r')

file.close()


# plot figure
#pp.figure(figsize=(8,5))
#pp.subplot(1,2,1); pp.imshow(monalisa_bw, cmap='gray')
#pp.subplot(1,2,2); pp.imshow(monalisa_xgrad, cmap='gray')
#







