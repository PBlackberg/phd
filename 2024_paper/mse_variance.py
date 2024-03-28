



def get_mse_anom2(switch, var_name, dataset, experiment, resolution, timescale):
    '''# MSE variance from the tropical mean  '''
    c_p, L_v = dD.dims_class.c_p, dD.dims_class.L_v
    ta, zg, hus = [vB.load_variable({var: True}, switch, dataset, experiment, resolution, timescale) for var in ['ta', 'zg', 'hus']]
    da = c_p * ta + zg + L_v * hus
    da, _ = pick_vert_reg(switch, dataset, da)
    da_sMean = mean_calc.get_sMean(da)
    da_anom = da - da_sMean
    da = da_anom**2
    return da


