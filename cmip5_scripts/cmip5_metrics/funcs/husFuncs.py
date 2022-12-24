import xarray as xr
import numpy as np
from os.path import expanduser
home = expanduser("~")



def calc_hus_tmean(hus):
    hus_tMean = hus.mean(dim='time', keep_attrs=True)

    hus_tMean = xr.Dataset(
        data = {'tas_tMean': hus_tMean}
                )

    return hus_tMean



def calc_hus_sMean(hus):
    aWeights = np.cos(np.deg2rad(hus.lat))
    hus_sMean= hus.weighted(aWeights).mean(dim=('lat','lon'))
    
    hus_sMean = xr.Dataset(
    data = {'hus_sMean': hus_sMean}
            )

    return hus_sMean








if __name__ == '__main__':


    from vars.myFuncs import *
    from vars.husVars import *


    models = [
            # 'IPSL-CM5A-MR', # 1
            'GFDL-CM3',     
            # 'GISS-E2-H',    # 3
            # 'bcc-csm1-1',   # 4
            # 'CNRM-CM5',     # 5
            # 'CCSM4',        # 6 # cannot concatanate files for historical run
            # 'HadGEM2-AO',   # 7
            # 'BNU-ESM',      # 8
            # 'EC-EARTH',     # 9
            # 'FGOALS-g2',    # 10
            # 'MPI-ESM-MR',   # 11
            # 'CMCC-CM',      # 12
            # 'inmcm4',       # 13
            # 'NorESM1-M',    # 14
            # 'CanESM2',      # 15 # slicing with .sel does not work, 'contains no datetime objects'
            # 'MIROC5',       # 16
            # 'HadGEM2-CC',   # 17
            # 'MRI-CGCM3',    # 18
            # 'CESM1-BGC'     # 19
            ]
    
    experiments = [
                #'historical',
                'rcp85'
                ]


    switch = {
        'local_files': True, 
        'nci_files': False, 
    }


    for model in models:
        for experiment in experiments:


            if switch['local_files']:
                folder = home + '/Documents/data/cmip5/' + model
                fileName = model + '_hus_' + experiment + '.nc'
                path = folder + '/' + fileName
                ds = xr.open_dataset(path)
                hus = ds.hus


            if switch['nci_files']:
                hus = get_hus(model, experiment).hus # from husVars
                folder = '/g/data/k10/cb4968/data/cmip5/'+ model



            hus_tMean = calc_hus_tmean(hus)


            hus_sMean = calc_hus_sMean(hus)




            saveit = False            
            if saveit:                
                dataSet = hus_tMean
                save_file(dataSet, folder, fileName)



            saveit = False            
            if saveit:                
                dataSet = hus_sMean
                save_file(dataSet, folder, fileName)