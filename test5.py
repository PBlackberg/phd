




# def load_data(switch, source, dataset, experiment, var):
#     ...
#     elif var == 'stability':    
#         da = gD.get_var_data(source, dataset, experiment, 'ta', switch) # Temperature at pressure levels (K)
#         nan_mask = da.isnull().any(dim='plev')                          # Find columns where there are NaNs
#         da = da.where(~nan_mask, np.nan)                                # Exclude columns with NaN
#         theta =  da * (1000e2 / da.plev)**(287/1005)                    # theta = T (P_0/P)^(R_d/C_p)
#         plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
#         da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
#         da = ((da1 * da1.plev).sum(dim='plev') / da1.plev.sum(dim='plev')) - ((da2 * da2.plev).sum(dim='plev') / da2.plev.sum(dim='plev'))   
#     else:
#         da = gD.get_var_data(source, dataset, experiment, var, switch)
#     return da

# def pick_hor_reg(switch, source, dataset, experiment, da):
#     ''' Pick out data in regions of ascent/descent based on 500 hPa vertical pressure velocity (wap)'''
#     region = ''
#     if switch['descent']:
#         wap500 = load_data(switch, source, dataset, experiment, 'wap').sel(plev = 500e2)
#         da = da.where(wap500>0)
#         region = '_d'
#     if switch['ascent']:
#         wap500 = load_data(switch, source, dataset, experiment, 'wap').sel(plev = 500e2)
#         da = da.where(wap500<0)
#         region = '_a'
#     return da, region

# def calc_sMean(da):
#     aWeights = np.cos(np.deg2rad(da.lat))
#     sMean = da.weighted(aWeights).mean(dim=('lat','lon'), keep_attrs=True).compute()
#     return sMean

# da =           load_data(switch, source, dataset, experiment, var)
# da, hor_reg =  pick_hor_reg(switch, source, dataset, experiment, da)
# da =           calc_sMean(da)









# da = gD.get_var_data(source, dataset, experiment, 'ta', switch)         # Temperature at pressure levels (K)
# theta =  da * (1000e2 / da['plev'])**(287/1005) 
# plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]
# da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))]
# w1, w2 = ~np.isnan(da1) * da1['plev'], ~np.isnan(da2) * da2['plev']     # Where there are no values, exclude the associated pressure levels from the weights
# weighted_mean = ((da1 * w1).sum() / w1.sum(dim='plev')) - ((da2 * w2).sum() / w2.sum(dim='plev'))





import numpy as np

a = np.nan
b= 5

c = b/a

print(c)


