# for cancat files:
# for f in files:  # one model from warming scenario from cmip5 have a file that needs to be removed (creates duplicate data otherwise)
#     files.remove(f) if f[f.index(".nc")-fileYear1_charStart : f.index(".nc")-fileYear1_charEnd]=='19790101' and f[f.index(".nc")-fileYear2_charStart : f.index(".nc")]=='20051231' else None





# settings for plot with individual numbers on xaxis for trend plot
        # mF.move_row(ax, 0.0875 - 0.025 +0.025) if row == 0 else None
        # mF.move_row(ax, 0.0495 - 0.0135+0.025) if row == 1 else None
        # mF.move_row(ax, 0.01   - 0.005+0.025)  if row == 2 else None
        # mF.move_row(ax, -0.0195+0.025)         if row == 3 else None
        # mF.move_row(ax, -0.05+0.025)           if row == 4 else None
        # mF.move_row(ax, -0.05+0.025)           if row == 5 else None



# creating a mask for excluding imcomplete sections
# da = gD.get_var_data(source, dataset, experiment, 'ta', switch)
# nan_mask = da.isnull().any(dim='plev')                        
# da = da.where(~nan_mask, np.nan)                              # When calculating the difference between two sections I exclude gridpoints where the value at any included pressure level is NaN
# theta =  da * (1000e2 / da.plev)**(287/1005)                  # theta = T (P_0/P)^(R_d/C_p)
# plevs1, plevs2 = [400e2, 250e2], [925e2, 700e2]               # pressure levels in ERA are reversed to cmip
# da1, da2 = [theta.sel(plev=slice(plevs1[0], plevs1[1])), theta.sel(plev=slice(plevs2[0], plevs2[1]))] if not dataset == 'ERA5' else [theta.sel(plev=slice(plevs1[1], plevs1[0])), theta.sel(plev=slice(plevs2[1], plevs2[0]))] 
# da = ((da1 * da1.plev).sum(dim='plev') / da1.plev.sum(dim='plev')) - ((da2 * da2.plev).sum(dim='plev') / da2.plev.sum(dim='plev'))   
