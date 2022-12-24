import xarray as xr
import numpy as np
from os.path import expanduser
home = expanduser("~")



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






if __name__ == '__main__':


    from vars.tasVars import *
    from vars.myFuncs import *


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



    switch = {
        'local_files': True, 
        'nci_files': False, 
    }


    for model in models:
        for experiment in experiments:


            if switch['local_files']:
                folder = home + '/Documents/data/cmip5/' + model
                fileName = model + '_tas_' + experiment + '.nc'
                path = folder + '/' + fileName
                ds = xr.open_dataset(path)
                tas = ds.tas


            if switch['nci_files']:
                tas = get_tas(model, experiment).tas # from husVars
                folder = '/g/data/k10/cb4968/data/cmip5/'+ model




            tas_tMean = calc_tas_tmean(tas)


            tas_sMean= calc_tas_sMean(tas)





            saveit = False            
            if saveit:                
                dataSet = tas_tMean
                save_file(dataSet, folder, fileName)


            saveit = False            
            if saveit:                
                dataSet = tas_sMean
                save_file(dataSet, folder, fileName)





