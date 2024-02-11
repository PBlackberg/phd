'''
Testing intake module for cmip

'''



import intake
cat = intake.cat
nci = cat['nci']
esgf = nci['esgf']
cmip6 = esgf.cmip6

values_dict = cmip6.unique()
# print(values_dict['activity_id'])

# subset = cmip6.search(activity_id = 'ScenarioMIP', experiment_id = 'ssp585', source_id = 'NorESM2-MM', table_id = 'Amon', variable_id = 'wap')
# print(subset.df)
# print(subset.df['path'][0])
# ds_dict = subset.to_dataset_dict()
# print(ds_dict)

print(cat['CMIP6.ScenarioMIP.NCC.NorESM2-MM.ssp585.r1i1p1f1.Amon.wap.gn.v20191108'].df)





















