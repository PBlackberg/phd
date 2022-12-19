import xarray as xr
import numpy as np

from vars.hus_vars import *
from vars.myFuncs import *



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



def calc_hus_annual(hus):
    aWeights = np.cos(np.deg2rad(hus.lat))
    hus_annual= hus.resample(time='Y').mean(dim='time', keep_attrs=True).weighted(aWeights).mean(dim=('lat','lon'))
    
    return hus_annual







if __name__ == '__main__':

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


    for model in models:
        for experiment in experiments:

            haveData = False
            if haveData:
                folder = '/g/data/k10/cb4968/data/cmip5/ds'
                fileName = model + '_hus_' + experiment + '.nc'
                path = folder + '/' + fileName
                precip = xr.open_dataset(path).hus
            else:
                hus = get_hus(model, experiment).hus




            hus_tMean = hus.mean(dim='time', keep_attrs=True)

            saveit = False            
            if saveit:                
                dataSet = hus_tMean
                myFuncs.save_file(dataSet, folder, fileName)







            aWeights = np.cos(np.deg2rad(hus.lat))
            hus_sMean= hus.weighted(aWeights).mean(dim=('lat','lon'))

            saveit = False            
            if saveit:                
                dataSet = hus_sMean
                myFuncs.save_file(dataSet, folder, fileName)






            hus_annual= hus.resample(time='Y').mean(dim='time', keep_attrs=True).weighted(aWeights).mean(dim=('lat','lon'))

            saveit = False            
            if saveit:                
                dataSet = hus_annual
                myFuncs.save_file(dataSet, folder, fileName)





