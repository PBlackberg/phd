import xarray as xr
import numpy as np
import timeit
from os.path import expanduser
home = expanduser("~")
from funcs.vars.myFuncs import *



# ------------------------------- select datasets to use and datafiles to generate ---------------------------------------------


models_cmip5 = [
        # 'IPSL-CM5A-MR', # 1
         'GFDL-CM3',     # 2
        # 'GISS-E2-H',    # 3
        # 'bcc-csm1-1',   # 4
        # 'CNRM-CM5',     # 5
        # #'CCSM4',        # 6 # cannot concatanate files for historical run
        # 'HadGEM2-AO',   # 7
        # 'BNU-ESM',      # 8
        # 'EC-EARTH',     # 9
        # 'FGOALS-g2',    # 10
        # 'MPI-ESM-MR',   # 11
        # 'CMCC-CM',      # 12
        # 'inmcm4',       # 13
        # 'NorESM1-M',    # 14
        # 'CanESM2',      # 15
        # 'MIROC5',       # 16
        # 'HadGEM2-CC',   # 17
        # 'MRI-CGCM3',    # 18
        # 'CESM1-BGC'     # 19
        ]

experiments = [
                'historical', 
                # 'rcp85'
            ]


dataFiles = {
                    'pr_rxday': True, 
                    'pr_percentiles': False,
                    'numberIndex': False, 
                    'pwad': False, 
                    'rome': False, 
                    'rome_n': False,

                    'tas_tMean': False,
                    'tas_sMean': False,
                    
                    'hus_tMean': False,
                    'hus_sMean': False
    }


switch = {
    'local_files': True, 
    'nci_files': False, 
}



for model in models:
    start = timeit.default_timer()
    
    for experiment in experiments:



# ------------------------------- Calculate metrics and save files ---------------------------------------------
            
        if (dataFiles['pr_rx1day'] or dataFiles['pr_percentiles'] or dataFiles['NumberIndex'] 
            or dataFiles['pwad'] or dataFiles['rome'] or dataFiles['rome_n']):

            if switch['local_files']:
                folder = home + '/Documents/data/cmip5/ds'
                fileName = model + '_precip_' + experiment + '.nc'
                path = folder + '/' + fileName
                ds = xr.open_dataset(path)
                precip = ds.precip*60*60*24
                precip.attrs['units']= 'mm/day'
                folder = home + '/Documents/data/cmip5/' + model

            if switch['nci_files']:
                from funcs.vars.prVars import *
                precip = get_pr(model, experiment).precip
                folder = '/g/data/k10/cb4968/data/cmip5/'+ model



            from funcs.prFuncs import *
            if dataFiles['pr_rxday']:
                fileName = model + '_pr_rxday_' + experiment + '.nc'
                dataSet = calc_rxday(precip)
                save_file(dataSet, folder, fileName)


            if dataFiles['pr_percentiles']:
                fileName = model + '_pr_percentiles_' + experiment + '.nc'
                dataSet = calc_pr_percentiles(precip)
                save_file(dataSet, folder, fileName)



            from funcs.aggFuncs import *
            if (dataFiles['numberIndex'] or dataFiles['pwad'] or dataFiles['rome'] or dataFiles['rome_n']):
                conv_threshold = precip.quantile(0.97,dim=('lat','lon'),keep_attrs=True).mean(dim='time',keep_attrs=True)
                

            if dataFiles['numberIndex']:
                fileName = model + '_numberIndex_' + experiment + '.nc'
                dataset = calc_numberIndex(precip, conv_threshold)
                save_file(dataset, folder, fileName) 


            if dataFiles['pwad']:
                fileName = model + '_pwad_' + experiment + '.nc'
                dataset = calc_pwad(precip, conv_threshold)
                save_file(dataset, folder, fileName) 


            if dataFiles['rome']:
                rome = calc_rome(precip, conv_threshold)
            else:
                rome = np.ones(len(precip.time.data))*np.nan

            if dataFiles['rome_n']:
                n = 8
                rome_n = calc_rome_n(n, precip, conv_threshold)
            else:
                rome_n = np.ones(len(precip.time.data))*np.nan


            if (dataFiles['rome'] or dataFiles['rome_n']):
                fileName = model + '_rome_' + experiment + '.nc'              
                dataset = xr.Dataset(
                    data_vars = {'rome': rome, 
                                 'rome_n':rome_n},
                    attrs = {'description': 'ROME based on all and the {} largest contiguous convective regions in the scene for each day'.format(n),
                             'units':'km^2'}                  
                        )
                save_file(dataset, folder, fileName)






        
        if (dataFiles['tas_tMean'] or dataFiles['tas_sMean']):
            
            if switch['local_files']:
                folder = home + '/Documents/data/cmip5/ds'
                fileName = model + '_tas_' + experiment + '.nc'
                path = folder + '/' + fileName
                ds = xr.open_dataset(path)
                tas = ds.tas
                folder = home + '/Documents/data/cmip5/' + model


            if switch['nci_files']:
                from funcs.vars.tasVars import *
                tas = get_tas(model, experiment).tas
                folder = '/g/data/k10/cb4968/data/cmip5/' + model


            from funcs.tasFuncs import *
            if dataFiles['tas_tMean']:
                fileName = model + '_tas_tMean_' + experiment + '.nc'
                dataSet = calc_tas_tMean(tas)
                save_file(dataSet, folder, fileName)


            if dataFiles['tas_sMean']:                
                fileName = model + '_tas_sMean_' + experiment + '.nc'     
                dataSet = calc_tas_sMean(tas)
                save_file(dataSet, folder, fileName)






        if (dataFiles['hus_tMean'] or dataFiles['hus_sMean']):

            if switch['local_files']:
                folder = home + '/Documents/data/cmip5/ds'
                fileName = model + '_hus_' + experiment + '.nc'
                path = folder + '/' + fileName
                ds = xr.open_dataset(path)
                hus = ds.hus
                folder = home + '/Documents/data/cmip5/' + model

            if switch['nci_files']:
                from funcs.vars.husVars import *
                hus = get_hus(model, experiment).hus
                folder = '/g/data/k10/cb4968/data/cmip5/'+ model



            from funcs.husFuncs import *
            if dataFiles['hus_tMean']:
                fileName = model + '_hus_tMean_' + experiment + '.nc'
                dataSet = calc_hus_tmean(hus)
                save_file(dataSet, folder, fileName)


            if dataFiles['hus_sMean']:
                fileName = model + '_hus_sMean' + experiment + '.nc'
                dataSet = calc_hus_sMean(hus)
                save_file(dataSet, folder, fileName)



    stop = timeit.default_timer()
    print('model: {} took {} minutes to finsih'.format(model, (stop-start)/60))

















