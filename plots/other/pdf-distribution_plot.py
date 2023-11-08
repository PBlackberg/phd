        # else:
        #     metric_class.color = 'k'
        #     if dataset in ['INM-CM5-0', 'CanESM5']:
        #         metric_class.color = 'y'


# # ---------------------------------------------------------------------------------------- load list ----------------------------------------------------------------------------------------------------- #
# def get_list(switchM, dataset, metric_class):
#     source = mV.find_source(dataset, mV.models_cmip5, mV.models_cmip6, mV.observations)
#     timescale = 'daily' if metric_class.var_type in ['pr', 'org', 'hus', 'ws'] else 'monthly'
#     experiment  = '' if source == 'obs' else mV.experiments[0]
#     # dataset = 'GPCP_1998-2009' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset
#     dataset = 'GPCP_2010-2022' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset # pick a time range
#     # dataset = 'GPCP' if source == 'obs' and metric_class.var_type in ['pr', 'org'] else dataset # complete record

#     alist = mF.load_metric(metric_class, mV.folder_save[0], source, dataset, timescale, experiment, mV.resolutions[0])[metric_class.name]
#     alist = mF.resample_timeMean(alist, mV.timescales[0])
#     axtitle = dataset
    
#     if dataset == 'CERES': # this observational dataset have monthly data with day specified as the middle of the month instead of the first
#         alist['time'] = alist['time'] - pd.Timedelta(days=14)

#     metric_title = ''

#     if switchM['anomalies']:
#         metric_title = 'anomalies'
#         if mV.timescales[0] == 'daily': 
#             rolling_mean = alist.rolling(time=12, center=True).mean()
#             alist = alist - rolling_mean
#             alist = alist.dropna(dim='time')
#         if mV.timescales[0] == 'monthly': 
#             climatology = alist.groupby('time.month').mean('time')
#             alist = alist.groupby('time.month') - climatology 
#         if mV.timescales[0] == 'annual':
#             '' 
#     return alist, metric_title, axtitle