import numpy as np
import os



def save_file(dataset, folder, fileName):
    
    os.makedirs(folder, exist_ok=True)
    path = folder + '/' + fileName
    if os.path.exists(path):
        os.remove(path)    
    dataset.to_netcdf(path, encoding=dataset.encoding.update({'zlib': True, 'complevel': 4}))





