'''
# ------------------------
#     Missing data
# ------------------------
Loop over models and skip datasets with no data
'''


# -------------------------------------------------------------------------------------- Packages --------------------------------------------------------------------------------------------------------- #
import numpy as np


# ----------------------------------------------------------------------------------- imported scripts --------------------------------------------------------------------------------------------------- #
import os
import sys
home = os.path.expanduser("~")

sys.path.insert(0, f'{os.getcwd()}/util-core')
import choose_datasets as cD                            # Chosen datasets



# ------------------------
#  Identify missing data
# ------------------------
# ----------------------------------------------------------------------------- mossing data lists --------------------------------------------------------------------------------------------------- #
def valid_source_experiment(source, experiment):
    valid_combinations = {
        ('test', 'historical'),
        ('cmip5', 'historical'), ('cmip5', 'rcp85'),
        ('cmip6', 'historical'), ('cmip6', 'ssp585'),
        ('dyamond', 'historical'),
        ('nextgems', 'historical'),
        ('obs', 'obs'),
        }
    return (source, experiment) in valid_combinations   # returns True if condition is met
    
def has_cl_data(var, dataset):
    if var in ['lcf', 'hcf', 'cl', 'ds_cl', 'cl_p_hybrid', 'p_hybrid'] \
        and dataset in [
                'INM-CM5-0', 
                'KIOST-ESM', 
                'EC-Earth3', 
                'INM-CM4-8', 
                'CNRM-CM6-1-HR', 
                'GFDL-ESM4'
                ]:                                 
        return False    
    return True
    
def has_pe_data(var, dataset):
    if var in ['pe'] \
        and dataset in [
                'MIROC-ES2L',        # 10  # no pe
                'ACCESS-ESM1-5',     # 18  # no pe
                'CNRM-ESM2-1',       # 19  # no pe
                'EC-Earth3',         # 20  # pe test (no pe anymore)
                'CNRM-CM6-1',        # 21  # no pe
                'CNRM-CM6-1-HR',     # 22  # no pe
                'ACCESS-CM2',        # 25  # no pe
                'UKESM1-0-LL',       # 29  # no pe
                'IITM-ESM'           # not missing but unrealistic percentiles
                ]:                                 
        return False    
    return True
    

# ----------------------------------------------------------------------------- Checking available data --------------------------------------------------------------------------------------------------- #
def data_available(var = '', source = '', dataset = '', experiment = ''):
    ''' Checks if dataset has variable and dataset-experiment combination makes sense'''
    if experiment and not valid_source_experiment(source, experiment):
        print(f'{dataset} - {experiment} is invalud combination (skipped)')
        return False
    if not has_cl_data(var, dataset):
        print(f'No {var} data for this dataset (skipped)')
        return False
    if not has_pe_data(var, dataset):
        print(f'No {var} data for this dataset (skipped)')
        return False
    return True


# ---------------------------------------------------------------------------------- loop and check --------------------------------------------------------------------------------------------------- #
def find_source(dataset):
    '''Determining source of dataset '''
    source = 'test'     if np.isin(cD.test_fields, dataset).any()      else None     
    source = 'cmip5'    if np.isin(cD.models_cmip5, dataset).any()     else source      
    source = 'cmip6'    if np.isin(cD.models_cmip6, dataset).any()     else source         
    source = 'dyamond'  if np.isin(cD.models_dyamond, dataset).any()   else source        
    source = 'nextgems' if np.isin(cD.models_nextgems, dataset).any()  else source   
    source = 'obs'      if np.isin(cD.observations, dataset).any()     else source
    return source

def run_experiment(var = '', dataset = '', experiments = ['a', 'b'], source = ''):
    for experiment in experiments:
        if not data_available(var, source, dataset, experiment):
            continue
        print(f'\t\t {dataset} ({source}) {experiment}') if experiment else print(f'\t {dataset} ({source}) observational dataset')
        yield dataset, experiment
        
def run_dataset(var = '', datasets = ['a', 'b'], experiments = ['a', 'b']):
    for dataset in datasets:
        source = find_source(dataset)
        print(f'\t{dataset} ({source})')
        yield from run_experiment(var, dataset, experiments, source)

def run_dataset_only(var = '', datasets = ['a', 'b']):
    for dataset in datasets:
        source = find_source(dataset)
        if not data_available(var, source, dataset):
            continue
        yield dataset


# ------------------------
#         Test
# ------------------------
if __name__ == '__main__':

    var = 'pe'

    switch = {
        'dataset_experiment': False,
        'dataset':            True,
        }

    if switch['dataset_experiment']:
        for dataset, experiment in run_dataset(var, cD.datasets, cD.experiments):
            print(f'\t\t\t\t {dataset} executes')

    if switch['dataset']:
        for dataset in run_dataset_only(var, cD.datasets):
            print('executes')

































