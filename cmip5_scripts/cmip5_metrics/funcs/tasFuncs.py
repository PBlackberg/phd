import xarray as xr
import numpy as np

from vars.tas_vars import *
from vars.myFuncs import *



def calc_tas_tmean(tas):
    tas_tMean = tas.mean(dim='time', keep_attrs=True)

    tas_tMean = xr.Dataset(
        data = {'tas_tMean': tas_tMean}
                )

    return tas_tMean



def calc_tas_sMean(tas):
    aWeights = np.cos(np.deg2rad(tas.lat))
    tas_sMean= tas.weighted(aWeights).mean(dim=('lat','lon'))
    
    tas_sMean = xr.Dataset(
    data = {'tas_sMean': tas_sMean}
            )

    return tas_sMean



def calc_tas_annual(tas):
    aWeights = np.cos(np.deg2rad(tas.lat))
    tas_annual= tas.resample(time='Y').mean(dim='time', keep_attrs=True).weighted(aWeights).mean(dim=('lat','lon'))
    
    return tas_annual







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
                'historical',
                # 'rcp85'
                ]


    for model in models:
        for experiment in experiments:

            haveData = False
            if haveData:
                folder = '/g/data/k10/cb4968/data/cmip5/ds'
                fileName = model + '_tas_' + experiment + '.nc'
                path = folder + '/' + fileName
                precip = xr.open_dataset(path).tas
            else:
                tas = get_tas(model, experiment).tas






            tas_tMean = tas.mean(dim='time', keep_attrs=True)

            saveit = False            
            if saveit:                
                dataSet = tas_tMean
                myFuncs.save_file(dataSet, folder, fileName)






            aWeights = np.cos(np.deg2rad(tas.lat))
            tas_sMean= tas.weighted(aWeights).mean(dim=('lat','lon'))

            saveit = False            
            if saveit:                
                dataSet = tas_sMean
                myFuncs.save_file(dataSet, folder, fileName)





            tas_annual= tas.resample(time='Y').mean(dim='time', keep_attrs=True).weighted(aWeights).mean(dim=('lat','lon'))

            saveit = False            
            if saveit:                
                dataSet = tas_annual
                myFuncs.save_file(dataSet, folder, fileName)







