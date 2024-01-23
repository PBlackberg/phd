def load_variable(switch = {'constructed_fields': False, 'sample_data': True}, var = 'pr', 
                    dataset = 'random', experiment = mV.experiments[0], timescale = mV.timescales[0]):
    ''' Loading variable data.
        Sometimes sections of years of a dataset will be used instead of the full data ex: if dataset = GPCP_1998-2010 (for obsservations) 
        (There is a double trend in high percentile precipitation rate for the first 12 years of the data (that affects the area picked out by the time-mean percentile threshold)'''
    source = find_source(dataset)
    dataset_alt = dataset.split('_')[0] if '_' in dataset and source in ['obs'] else dataset # sometimes a year_range version of the obs will be used ex: 'GPCP_2010-2022'   
    if source == 'test': # for testing script
        var = 'pr'  if var == 'var_2d' else var
        var = 'hur' if var == 'var_3d' else var 
                                                   
    da = cF.get_cF_var(dataset_alt, var)                                                                                                                    if switch['constructed_fields']             else None
    da = xr.open_dataset(f'{mV.folder_save[0]}/sample_data/{var}/{source}/{dataset_alt}_{var}_{timescale}_{experiment}_{mV.resolutions[0]}.nc')[f'{var}']   if switch['sample_data']                    else da  
    da = gD.get_var_data(source, dataset_alt, experiment, var)                                                                                              if switch['gadi_data']                      else da
    if '_' in dataset:                                                  
        start_year, end_year = dataset.split('_')[1].split('-')
        da = da.sel(time= slice(start_year, end_year))
    return da


