def find_list_source(datasets, models_cmip5 = mV.models_cmip5, models_cmip6 = mV.models_cmip6, observations = mV.observations):
    ''' Determining source of dataset list (for plots) '''
    sources = set()
    for dataset in datasets:
        sources.add('cmip5') if dataset in models_cmip5 else None
        sources.add('cmip6') if dataset in models_cmip6 else None
        sources.add('obs')   if dataset in observations else None
    list_source = 'cmip5' if 'cmip5' in sources else 'test'
    list_source = 'cmip6' if 'cmip6' in sources else list_source
    list_source = 'obs'   if 'obs'   in sources else list_source
    list_source = 'mixed' if 'cmip5' in sources and 'cmip6' in sources else list_source
    return list_source

def find_ifWithObs(datasets, observations= mV.observations):
    ''' Indicate if there is observations in the dataset list (for plots) '''
    for dataset in datasets:
        if dataset in observations:
            return '_withObs'
    return ''